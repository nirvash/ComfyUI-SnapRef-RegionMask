"""SnapRef Region Probe / Auto Mask: find where Anima draws person 1 / person 2 and turn it into masks.

Records the cross-attention of Anima (MiniTrainDIT) image tokens to the prompt tokens of two person terms
(e.g. "1girl" and "1man") during sampling. SnapRef Region Auto Mask turns a recording plus the decoded image into
mask_1 / mask_2 for SnapRef Region Mask, so the references can be placed without drawing masks by hand.

The probe does not change the sampling result: it only reads q / k of the cross-attention.

Nodes
  SnapRefRegionProbe          MODEL + CLIP + prompt + two person terms -> MODEL, REGION_PROBE. Last in the model chain.
  SnapRefRegionAutoMask       IMAGE + REGION_PROBE -> mask_1, mask_2 (learned per-block weights, token-feature parts,
                              superpixel vote; runs on the sampling GPU).
  SnapRefRegionAutoMaskPanel  the estimation image and the final image with the regions drawn over them.
  SnapRefRegionProbeOverlay   analysis: attention heat map and plain split of one recording.
  SnapRefSubsampleSigmas      analysis: finish a schedule in fewer steps.
"""
import json
import logging
import math
import os

import torch
import torch.nn.functional as F

import comfy.patcher_extension

__version__ = "0.1.0"

KEY = "snapref_region_probe"
DEFAULT_WINDOWS = (0.1, 0.2, 0.3, 1.0)


# ---------------------------------------------------------------- token positions

def find_subsequence(seq, sub):
    """All start indices where list `sub` occurs in list `seq`."""
    if not sub:
        return []
    n = len(sub)
    return [i for i in range(len(seq) - n + 1) if seq[i:i + n] == sub]


def t5_ids(clip, text):
    """T5 token ids of `text` as Anima's CLIPTextEncode sees them (the cross-attention keys follow them 1:1)."""
    tokens = clip.tokenize(text)
    if "t5xxl" not in tokens:
        raise ValueError("[SnapRefRegionProbe] CLIP has no t5xxl tokens; this probe is for Anima")
    return [int(t[0]) for t in tokens["t5xxl"][0]]


def term_positions(prompt_ids, term_ids_list, end_id=1):
    """Positions of every occurrence of any term variant. term_ids_list: one id list per variant."""
    found = set()
    for ids in term_ids_list:
        ids = [i for i in ids if i != end_id]
        for start in find_subsequence(prompt_ids, ids):
            found.update(range(start, start + len(ids)))
    return sorted(found)


def split_terms(text):
    """'1girl | girl' -> ['1girl', 'girl'] (variants separated by '|')."""
    return [part.strip() for part in text.split("|") if part.strip()]


# ---------------------------------------------------------------- recording

def sampling_progress(options):
    """(step index, fraction done) of the sampling schedule, or (None, None) if unknown."""
    sig, sched = options.get("sigmas"), options.get("sample_sigmas")
    if sig is None or sched is None or len(sched) < 2:
        return None, None
    s = float(sig.flatten()[0])
    sched = [float(v) for v in sched.flatten()[:-1]]   # the last sigma (0) is never evaluated
    # nearest step: the evaluated sigma can sit slightly below the schedule value
    idx = min(range(len(sched)), key=lambda i: abs(sched[i] - s))
    return idx, idx / len(sched)


def cross_attention_share(q, k, positions, n_keys=None):
    """Share of the softmax attention each image token gives to each token group.

    q [B, S, H, D], k [B, N, H, D]; positions: list of index lists -> [len(positions), B, S] (mean over heads).
    n_keys: only the first n_keys keys take part in the softmax (Anima pads the prompt with zeros to 512 keys;
    the padding soaks up most of the attention and flattens the maps)."""
    if n_keys is not None:
        k = k[:, :n_keys]
    scale = q.shape[-1] ** -0.5
    out = []
    for b in range(q.shape[0]):
        logits = torch.einsum("shd,nhd->hsn", q[b].float(), k[b].float()) * scale
        probs = logits.softmax(dim=-1)
        out.append(torch.stack([probs[..., p].sum(-1).mean(0) for p in positions]))   # [G, S]
        del logits, probs
    return torch.stack(out, dim=1)


class Recorder:
    """Per-step sum of the attention share of each person term, over blocks and cond rows."""

    def __init__(self, terms, positions, n_keys=None, keep_blocks=False, keep_features=False,
                 feature_steps=None, feature_blocks=None):
        self.terms = terms
        self.positions = positions
        self.n_keys = n_keys
        self.keep_blocks = keep_blocks
        self.keep_features = keep_features
        self.feature_steps = set(FEATURE_STEPS if feature_steps is None else feature_steps)
        self.feature_blocks = set(FEATURE_BLOCKS if feature_blocks is None else feature_blocks)
        self.reset()

    def reset(self):
        self.steps = {}      # step index -> [G, S] float32 (cpu)
        self.counts = {}     # step index -> contributions summed
        self.blocks = {}     # (step index, block index) -> [G, S] float16 (cpu), for per-layer analysis
        self.features = {}   # (step index, block index) -> [S, FEATURE_DIM] float16: PCA of the token features
        self.progress = {}   # step index -> fraction
        self.grid = None     # (T, gh, gw)
        self.total_steps = None
        self.last_idx = None
        self.multi_row_warned = False

    def add(self, idx, share, block=None):
        share = share.detach().float().cpu()
        if idx in self.steps:
            self.steps[idx] += share
        else:
            self.steps[idx] = share.clone()
        self.counts[idx] = self.counts.get(idx, 0) + 1
        if block is not None:
            self.blocks[(idx, block)] = share.half()

    def add_feature(self, idx, block, x):
        """x [S, D] token features after self-attention (the cross-attention input): keep a PCA projection."""
        if idx not in self.feature_steps or block not in self.feature_blocks:
            return
        x = x.float()
        mean = x.mean(0, keepdim=True)
        q = min(FEATURE_DIM, *x.shape)
        _, _, v = torch.pca_lowrank(x - mean, q=q, center=False)
        self.features[(idx, block)] = ((x - mean) @ v[:, :q]).half().cpu()

    def dump(self, path):
        """Save per-step, per-block maps (and token features when recorded) for offline analysis (numpy .npz)."""
        import numpy as np
        steps = sorted(self.steps)
        nblocks = 1 + max((b for _, b in self.blocks), default=-1)
        G, S = self.steps[steps[0]].shape
        per_block = np.zeros((len(steps), nblocks, G, S), dtype=np.float16)
        for (i, b), m in self.blocks.items():
            per_block[steps.index(i), b] = m.numpy()
        extra = {}
        if self.features:
            f_steps = sorted({i for i, _ in self.features})
            f_blocks = sorted({b for _, b in self.features})
            feats = np.zeros((len(f_steps), len(f_blocks), S, FEATURE_DIM), dtype=np.float16)
            for (i, b), f in self.features.items():
                feats[f_steps.index(i), f_blocks.index(b)] = f.numpy()
            extra = {'features': feats, 'feature_steps': np.array(f_steps), 'feature_blocks': np.array(f_blocks)}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, per_block=per_block, steps=np.array(steps),
                            progress=np.array([self.progress[i] for i in steps]),
                            grid=np.array(self.grid), terms=np.array(self.terms), **extra)

    def window(self, end):
        """Mean map [G, S] over steps with progress < end (end >= 1 means every step), or None."""
        picked = [i for i in sorted(self.steps) if end >= 1.0 or self.progress[i] < end - 1e-9]
        return self.mean(picked), picked

    def mean(self, picked):
        if not picked:
            return None
        return sum(self.steps[i] / self.counts[i] for i in picked) / len(picked)


_STATE = {"recorder": None, "fingerprint": None}
# analysis dumps: token features (self-attention output) of these steps / blocks, PCA to FEATURE_DIM
FEATURE_STEPS = (3, 8, 12, 20, 28)
FEATURE_BLOCKS = (4, 8, 12, 16, 20, 24)
FEATURE_DIM = 64


def _attach(dm, recorder, idx):
    restore = []
    for block_index, block in enumerate(dm.blocks):
        attn = getattr(block, "cross_attn", None)
        if attn is None or not hasattr(attn, "compute_qkv"):
            continue
        previous = attn.forward

        def forward(x, context=None, rope_emb=None, transformer_options={}, _attn=attn, _prev=previous,
                    _b=block_index):
            fp = _STATE["fingerprint"]
            if fp is not None and context is not None:
                try:
                    rows = [r for r in range(context.shape[0])
                            if context[r].shape == fp.shape and torch.equal(context[r], fp.to(context))]
                except RuntimeError:
                    rows = []
                if rows:
                    if len(rows) > 1 and not recorder.multi_row_warned:
                        logging.warning("[SnapRefRegionProbe] %d cond rows; recording the first only (use batch_size 1)",
                                        len(rows))
                        recorder.multi_row_warned = True
                    with torch.no_grad():
                        r = rows[:1]
                        q, k, _ = _attn.compute_qkv(x[r], context[r], rope_emb=rope_emb,
                                                    transformer_options=transformer_options)
                        recorder.add(idx, cross_attention_share(q, k, recorder.positions, recorder.n_keys)[:, 0],
                                     block=_b if recorder.keep_blocks else None)
                        if recorder.keep_features:
                            recorder.add_feature(idx, _b, x[r][0])
            return _prev(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

        restore.append((attn, previous))
        attn.forward = forward
    return restore


def _detach(restore):
    for attn, previous in reversed(restore):
        attn.forward = previous


def cond_rows(flags, batch):
    """Row indices of the conditional chunks in a full cond/uncond batch, or None if the batch is split."""
    if not flags or batch % len(flags):
        return None
    per = batch // len(flags)
    return [i * per + j for i, f in enumerate(flags) if f == 0 for j in range(per)]


def _make_wrapper(recorder, step_offset=0, total_steps=0):
    """step_offset / total_steps: this sampler runs a segment of a longer schedule (SplitSigmas); record the
    steps under their index in the whole schedule."""
    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        dm = executor.class_obj
        opts = kwargs.get("transformer_options", {})
        idx, progress = sampling_progress(opts)
        if idx is not None and (step_offset or total_steps):
            idx += step_offset
            progress = idx / total_steps if total_steps else progress
        rows = cond_rows(opts.get("cond_or_uncond"), int(x.shape[0]))
        if idx is None or not rows or context is None:
            return executor(x, timesteps, context, *args, **kwargs)
        if recorder.last_idx is not None and idx < recorder.last_idx:
            recorder.reset()   # a new sampling run with the same (cached) model
        recorder.last_idx = idx
        recorder.total_steps = total_steps or len(opts.get("sample_sigmas")) - 1
        recorder.progress[idx] = progress
        p = int(getattr(dm, "patch_spatial", 2))
        T = x.shape[2] if x.ndim == 5 else 1
        recorder.grid = (T, math.ceil(x.shape[-2] / p), math.ceil(x.shape[-1] / p))
        _STATE["fingerprint"] = context[rows[0]].detach()
        restore = _attach(dm, recorder, idx)
        try:
            return executor(x, timesteps, context, *args, **kwargs)
        finally:
            _detach(restore)
            _STATE["fingerprint"] = None
    return wrapper


class SnapRefRegionProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "connect last: after SnapRef / SnapRef Region Mask"}),
            "clip": ("CLIP", {"tooltip": "the CLIP used for the positive prompt (Anima)"}),
            "text": ("STRING", {"multiline": True, "tooltip": "exactly the positive prompt text"}),
            "person_1": ("STRING", {"default": "1girl", "tooltip": "term(s) of person 1 (SnapRef image), "
                                                                   "variants separated by |"}),
            "person_2": ("STRING", {"default": "1boy", "tooltip": "term(s) of person 2 (SnapRef image_2)"}),
        }, "optional": {
            "block_weights": (["none"] + weight_names(), {
                "default": "none",
                "tooltip": "learned per-block weights for the person 1 / person 2 split (weights/*.json); "
                           "none = plain mean of all blocks"}),
            "dump_path": ("STRING", {"default": "", "tooltip": "analysis: save per-step, per-block maps to this .npz"}),
            "auto_mask": (["none"] + auto_weight_names(), {
                "default": "none",
                "tooltip": "record what SnapRef Region Auto Mask needs (per-block maps and token features) "
                           "for these weights (weights/*.json with probe_steps)"}),
            "continue_from": ("REGION_PROBE", {"tooltip": "segmented sampling: keep recording into this probe "
                                                          "(the previous segment's); its settings are used"}),
            "step_offset": ("INT", {"default": 0, "min": 0, "max": 10000,
                                    "tooltip": "segmented sampling: index of this segment's first step in the whole schedule"}),
            "total_steps": ("INT", {"default": 0, "min": 0, "max": 10000,
                                    "tooltip": "segmented sampling: steps of the whole schedule (0 = this sampler's)"}),
        }}

    RETURN_TYPES = ("MODEL", "REGION_PROBE")
    FUNCTION = "apply"
    CATEGORY = "SnapRef"
    DESCRIPTION = "Records where the image attends to the person 1 / person 2 terms during sampling."

    def apply(self, model, clip, text, person_1, person_2, block_weights="none", dump_path="", auto_mask="none",
              continue_from=None, step_offset=0, total_steps=0):
        if continue_from is not None:
            m = model.clone()
            self._wrap(m, _make_wrapper(continue_from["recorder"], step_offset, total_steps))
            return (m, {**continue_from, "dump_path": dump_path.strip()})
        weights = load_block_weights(block_weights)
        auto = load_auto_weights(auto_mask)
        prompt_ids = t5_ids(clip, text)
        terms, positions = [], []
        for term in (person_1, person_2):
            variants = split_terms(term)
            if not variants:
                raise ValueError("[SnapRefRegionProbe] empty person term")
            pos = term_positions(prompt_ids, [t5_ids(clip, v) for v in variants])
            if not pos:
                raise ValueError(f"[SnapRefRegionProbe] term {term!r} not found in the prompt tokens")
            terms.append(term)
            positions.append(pos)
        if set(positions[0]) & set(positions[1]):
            raise ValueError("[SnapRefRegionProbe] person_1 and person_2 share prompt tokens")
        dumping = bool(dump_path.strip())
        f_steps = set(FEATURE_STEPS) if dumping else set()
        f_blocks = set(FEATURE_BLOCKS) if dumping else set()
        if auto is not None:
            f_steps |= {e["feature_step"] for e in auto["probe_steps"].values()}
            f_blocks |= set(auto["feature_blocks"])
        recorder = Recorder(terms, positions, n_keys=len(prompt_ids),
                            keep_blocks=dumping or weights is not None or auto is not None,
                            keep_features=bool(f_steps), feature_steps=f_steps, feature_blocks=f_blocks)
        m = model.clone()
        self._wrap(m, _make_wrapper(recorder, step_offset, total_steps))
        return (m, {"recorder": recorder, "prompt_tokens": len(prompt_ids), "dump_path": dump_path.strip(),
                    "block_weights": block_weights if weights is not None else "none", "weights": weights,
                    "auto_mask": auto_mask if auto is not None else "none"})

    @staticmethod
    def _wrap(m, wrapper):
        dmw = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
        m.add_wrapper_with_key(dmw, KEY, wrapper)
        # outermost: sees the full cond/uncond batch before SnapRef splits it
        wrappers = m.wrappers.get(dmw, {})
        mine = wrappers.pop(KEY)
        m.wrappers[dmw] = {KEY: mine, **wrappers}


# ---------------------------------------------------------------- overlay

COLORS = (torch.tensor([1.0, 0.15, 0.15]), torch.tensor([0.15, 0.45, 1.0]))


def normalize(maps, low=0.02, high=0.995):
    """[G, h, w] -> each map stretched so its `low` quantile is 0 and its `high` quantile is 1."""
    flat = maps.flatten(1).float()
    lo = flat.quantile(low, dim=1).view(-1, 1, 1)
    hi = flat.quantile(high, dim=1).view(-1, 1, 1)
    return ((maps - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)


def otsu(values, bins=64):
    """Otsu threshold of values in 0..1."""
    hist = torch.histc(values.flatten().float(), bins=bins, min=0.0, max=1.0)
    p = hist / hist.sum().clamp_min(1)
    centers = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
    w0 = p.cumsum(0)
    m0 = (p * centers).cumsum(0)
    total = m0[-1]
    between = (total * w0 - m0) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    return float(centers[int(between[:-1].argmax())] + 0.5 / bins)


def assign(norm, threshold):
    """norm [G, h, w] in 0..1 -> (boolean masks [G, h, w], threshold used).

    Foreground: max over persons above `threshold` (0 = Otsu); split between the persons by argmax."""
    peak = norm.amax(dim=0)
    if threshold <= 0:
        threshold = otsu(peak)
    fg = peak > threshold
    label = norm.argmax(dim=0)
    return torch.stack([fg & (label == g) for g in range(norm.shape[0])]), threshold


def upsample(maps, h, w):
    return F.interpolate(maps[None].float(), size=(h, w), mode="bilinear", align_corners=False)[0]


def heat_panel(img, norm):
    out = img.clone() * 0.55
    for g in range(min(2, norm.shape[0])):
        a = (norm[g] * 0.85)[..., None]
        out = out * (1 - a) + COLORS[g] * a
    return out.clamp(0, 1)


def assign_panel(img, masks):
    out = img.clone() * 0.6
    for g in range(min(2, masks.shape[0])):
        a = (masks[g].float() * 0.55)[..., None]
        out = out * (1 - a) + COLORS[g] * a
    # 1-pixel outlines of each region
    for g in range(min(2, masks.shape[0])):
        m = masks[g].float()[None, None]
        edge = (F.max_pool2d(m, 3, 1, 1) - m)[0, 0] > 0
        out[edge] = COLORS[g]
    return out.clamp(0, 1)


def stats(masks, norm):
    total = masks[0].numel()
    a, b = masks[0].float().sum().item(), masks[1].float().sum().item()
    over = ((norm[0] > 0.5) & (norm[1] > 0.5)).float().sum().item()
    union = ((norm[0] > 0.5) | (norm[1] > 0.5)).float().sum().item()
    return {"area_1": round(a / total, 4), "area_2": round(b / total, 4),
            "overlap_iou_at_0.5": round(over / union, 4) if union else None}


def agreement(masks, ref_masks):
    """How well an A/B split matches a reference split, inside the reference foreground.

    accuracy: share of reference-foreground pixels given the same person; iou_g per person."""
    fg = ref_masks.any(dim=0)
    n = fg.float().sum().item()
    label = masks[1] & ~masks[0]                  # True = person 2 (argmax already makes them exclusive)
    ref_label = ref_masks[1]
    # pixels outside the estimate's foreground still count by argmax: background leak is acceptable
    acc = ((label == ref_label) & fg).float().sum().item() / n if n else None
    ious = []
    for g in range(ref_masks.shape[0]):
        inter = (masks[g] & ref_masks[g]).float().sum().item()
        union = (masks[g] | ref_masks[g]).float().sum().item()
        ious.append(round(inter / union, 4) if union else None)
    return {"accuracy": round(acc, 4) if acc is not None else None, "iou": ious}


def split_only(norm):
    """A/B split by argmax everywhere (no foreground cut): [G, h, w] booleans."""
    label = norm.argmax(dim=0)
    return torch.stack([label == g for g in range(norm.shape[0])])


STRIP_STEPS = (1, 2, 3, 4, 6, 8, 12)
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def _weight_files():
    out = {}
    try:
        names = os.listdir(WEIGHTS_DIR)
    except OSError:
        return out
    for f in names:
        if f.endswith(".json"):
            try:
                with open(os.path.join(WEIGHTS_DIR, f), encoding="utf-8") as fh:
                    out[f[:-5]] = json.load(fh)
            except (OSError, ValueError):
                continue
    return out


def weight_names():
    """Per-block weights for the Overlay split (json with "weights")."""
    return sorted(n for n, d in _weight_files().items() if "weights" in d)


def auto_weight_names():
    """Weights for SnapRef Region Auto Mask (json with "probe_steps")."""
    return sorted(n for n, d in _weight_files().items() if "probe_steps" in d)


def load_auto_weights(name):
    if not name or name == "none":
        return None
    with open(os.path.join(WEIGHTS_DIR, name + ".json"), encoding="utf-8") as f:
        data = json.load(f)
    if "probe_steps" not in data:
        raise ValueError(f"[SnapRefRegionProbe] {name} has no probe_steps (not auto mask weights)")
    return data


def load_block_weights(name):
    """{k: tensor [blocks + 1]} (last value = intercept), or None for "none"."""
    if not name or name == "none":
        return None
    with open(os.path.join(WEIGHTS_DIR, name + ".json"), encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): torch.tensor(v, dtype=torch.float32) for k, v in data["weights"].items()}


def pick_k(table, k):
    """The trained step count to use for k steps: the largest one <= k (the smallest if none)."""
    keys = sorted(table)
    below = [x for x in keys if x <= k]
    return below[-1] if below else keys[0]


def block_features(rec, picked, T, gh, gw, device=None):
    """[blocks, gh, gw]: per block, normalized person-1 map minus person-2 map, mean over `picked` steps."""
    nb = 1 + max(b for _, b in rec.blocks)
    maps = torch.stack([torch.stack([rec.blocks[(i, b)] for b in range(nb)]) for i in picked])
    maps = maps.to(device).float().mean(0) if device is not None else maps.float().mean(0)
    feats = []
    for b in range(nb):
        n = normalize(maps[b].view(-1, T, gh, gw)[:, 0])
        feats.append(n[0] - n[1])
    return torch.stack(feats)


def weighted_score(feats, w):
    """Score map [gh, gw]: > 0 means person 1."""
    return (feats * w[:-1].view(-1, 1, 1)).sum(0) + w[-1]


def score_split(score):
    return torch.stack([score > 0, score <= 0])


def _font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label(img, text):
    """Readable label in the top-left corner, sized to the panel."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    pil = Image.fromarray((img.numpy() * 255).astype("uint8"))
    draw = ImageDraw.Draw(pil)
    font = _font(max(12, pil.height // 28))
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    pad = max(4, (y1 - y0) // 3)
    draw.rectangle([0, 0, x1 + 2 * pad, y1 + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return torch.from_numpy(__import__("numpy").array(pil)).float() / 255


def caption_bar(width, text, height_hint):
    """White bar with wrapped caption lines, [h, width, 3]."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    import textwrap
    font = _font(max(14, height_hint // 36))
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    char_w = max(1, probe.textbbox((0, 0), "abcdefghij", font=font)[2] / 10)
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=max(20, int(width * 0.96 / char_w))) or [""]
    line_h = probe.textbbox((0, 0), "Ag", font=font)[3] + max(4, font.size // 3)
    pad = line_h // 2
    pil = Image.new("RGB", (width, pad * 2 + line_h * len(lines)), (255, 255, 255))
    draw = ImageDraw.Draw(pil)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(0, 0, 0), font=font)
    return torch.from_numpy(__import__("numpy").array(pil)).float() / 255


class SnapRefRegionProbeOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "probe": ("REGION_PROBE",),
            "window_end": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                                     "tooltip": "use the steps before this fraction of sampling (match ref_start)"}),
            "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "foreground: normalized attention above this; 0 = Otsu (automatic)"}),
        }, "optional": {
            "caption": ("STRING", {"multiline": True, "default": "",
                                   "tooltip": "text printed above the panel (seed, prompt, ...)"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("panel", "mask_1", "mask_2")
    FUNCTION = "apply"
    OUTPUT_NODE = True
    CATEGORY = "SnapRef"

    def apply(self, image, probe, window_end=0.1, threshold=0.0, caption=""):
        rec = probe["recorder"]
        img = image[0].float().cpu()
        H, W = img.shape[:2]
        if not rec.steps or rec.grid is None:
            raise RuntimeError("[SnapRefRegionProbe] nothing recorded; is the probe the last model node?")
        T, gh, gw = rec.grid
        weights = probe.get("weights")
        nb = 1 + max((b for _, b in rec.blocks), default=-1)
        if weights is not None and any(len(w) - 1 != nb for w in weights.values()):
            logging.warning("[SnapRefRegionProbe] block weights are for another block count (%d recorded); "
                            "using the plain mean", nb)
            weights = None
        weight_name = probe.get("block_weights", "none") if weights is not None else "none"
        info = {"version": __version__, "terms": rec.terms, "token_positions": rec.positions,
                "prompt_tokens": probe["prompt_tokens"], "grid": [gh, gw], "total_steps": rec.total_steps,
                "recorded_steps": sorted(rec.steps), "threshold": threshold, "window_end": window_end,
                "block_weights": weight_name, "windows": {}}

        def split_masks(picked, size=None):
            """Person 1 / person 2 split from the first steps `picked`: weighted blocks when available."""
            if weights is not None:
                score = weighted_score(block_features(rec, picked, T, gh, gw), weights[pick_k(weights, len(picked))])
                if size:
                    score = upsample(score[None], *size)[0]
                return score_split(score)
            norm = normalize(rec.mean(picked).view(-1, T, gh, gw)[:, 0])
            return split_only(upsample(norm, *size) if size else norm)

        def window_maps(end):
            mean, picked = rec.window(end)
            if mean is None:
                return None, picked
            return mean.view(-1, T, gh, gw)[:, 0], picked

        main, picked = window_maps(window_end)
        if main is None:
            raise RuntimeError(f"[SnapRefRegionProbe] no step before {window_end}")
        full, _ = window_maps(1.0)
        for end in sorted(set(DEFAULT_WINDOWS) | {window_end}):
            maps, steps = window_maps(end)
            if maps is None:
                continue
            norm = normalize(maps)
            masks, used = assign(norm, threshold)
            info["windows"][f"{end:g}"] = {
                "steps": steps, "threshold_used": round(used, 4), **stats(masks, norm),
                "maps": [[[round(float(v), 3) for v in row] for row in m] for m in norm]}

        # normalize and threshold on the token grid, then upsample (bilinear) to the image
        norm_main = normalize(main)
        _, used_main = assign(norm_main, threshold)
        norm_main = upsample(norm_main, H, W).clamp(0, 1)
        masks_main, _ = assign(norm_main, used_main)
        if weights is not None:   # foreground from the plain mean, person split from the weighted blocks
            masks_main = split_masks(picked, (H, W)) & masks_main.any(dim=0)
        norm_full = normalize(full)
        _, used_full = assign(norm_full, threshold)
        masks_full, _ = assign(upsample(norm_full, H, W).clamp(0, 1), used_full)
        steps_txt = f"steps {picked[0]}-{picked[-1]}" if picked else ""
        panels = [label(img, "generated"),
                  label(heat_panel(img, norm_main), f"attention <{window_end:g} {steps_txt}"),
                  label(assign_panel(img, masks_main), f"regions <{window_end:g} t={used_main:.2f} red=1 blue=2"),
                  label(assign_panel(img, masks_full), f"regions all steps t={used_full:.2f}")]
        rows = [torch.cat(panels, dim=1)]

        # how many first steps are needed: compare the split from the first k steps with the split of the
        # last third of the steps (the picture is settled there), on the token grid
        order = sorted(rec.steps)
        late = order[len(order) * 2 // 3:]
        late_norm = normalize(rec.mean(late).view(-1, T, gh, gw)[:, 0])
        ref_masks, ref_t = assign(late_norm, threshold)
        curve = []
        for k in range(1, len(order) + 1):
            norm_k = normalize(rec.mean(order[:k]).view(-1, T, gh, gw)[:, 0])
            plain = agreement(split_only(norm_k), ref_masks)
            point = {"k": k, "last_step": order[k - 1], **agreement(split_masks(order[:k]), ref_masks),
                     "accuracy_plain_mean": plain["accuracy"]}
            curve.append(point)
        info["step_curve"] = {"reference_steps": [late[0], late[-1]], "reference_threshold": round(ref_t, 4),
                              "split": weight_name, "points": curve}
        strip = []
        for k in [k for k in STRIP_STEPS if k <= len(order)] + [len(order)]:
            masks_k = split_masks(order[:k], (H, W))
            acc = curve[k - 1]["accuracy"]
            name = "all steps" if k == len(order) else f"first {k} step{'s' if k > 1 else ''}"
            strip.append(label(assign_panel(img, masks_k), f"{name}: agree {acc:.0%}" if acc is not None else name))
        while len(strip) % 4:
            strip.append(torch.zeros_like(img))
        rows += [torch.cat(strip[i:i + 4], dim=1) for i in range(0, len(strip), 4)]
        legend = (f"red = person 1 ({rec.terms[0]})   blue = person 2 ({rec.terms[1]})   "
                  f"split: {'block weights ' + weight_name if weights is not None else 'mean of all blocks'}   "
                  f"agree = match with the split of steps {late[0]}-{late[-1]} (mean of all blocks)")
        bar = caption_bar(rows[0].shape[1], (caption.strip() + "\n" if caption.strip() else "") + legend, H)
        if bar is not None:
            rows.insert(0, bar)
        panel = torch.cat(rows, dim=0)[None]

        if probe.get("dump_path"):
            rec.dump(probe["dump_path"])
            info["dump_path"] = probe["dump_path"]
        ui = {"snapref_probe_info": [json.dumps(info, ensure_ascii=False)]}
        return {"ui": ui, "result": (panel, masks_main[0].float()[None], masks_main[1].float()[None])}


# ---------------------------------------------------------------- auto mask (two-stage)
#
# Pass 1 (prompt + seed only, no SnapRef) is recorded by the probe; this turns it into mask_1 / mask_2:
#   score  = ridge weights over per-block (person 1 - person 2) maps of step windows of pass 1, per token
#   parts  = k-means over the model's own token features (blocks 8/12/16 of one step) + position
#   coarse = each part takes the sign of its summed score; fine = each SLIC superpixel of the pass-1 image
#            takes the majority of the coarse label. Every pixel goes to person 1 or person 2.
# Fitted and evaluated against hand-checked masks (29 images, see weights/hakushi_anima_gt_v1.json).

def kmeans(X, k, iters=25, seed=0):
    """k-means (k-means++ init, fixed seed) -> labels [N]."""
    X = torch.as_tensor(X).float()
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = [X[torch.randint(len(X), (1,), generator=g, device=X.device)].squeeze(0)]
    d = torch.cdist(X, c[0][None])[:, 0] ** 2
    for _ in range(k - 1):
        c.append(X[torch.multinomial(d / d.sum(), 1, generator=g)].squeeze(0))
        d = torch.minimum(d, torch.cdist(X, c[-1][None])[:, 0] ** 2)
    c = torch.stack(c)
    for _ in range(iters):
        lab = torch.cdist(X, c).argmin(1)
        new = torch.zeros_like(c).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=k).clamp_min(1).unsqueeze(1).float()
        new = new / cnt
        empty = torch.bincount(lab, minlength=k) == 0
        new[empty] = c[empty]
        if torch.allclose(new, c, atol=1e-4):
            break
        c = new
    return torch.cdist(X, c).argmin(1)


def token_parts(feats, gh, gw, k, position_weight):
    """feats: list of [S, D] token features -> part ids [gh, gw] (1..k)."""
    import numpy as np
    chunks = []
    for x in feats:
        x = x.float().numpy()
        chunks.append(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6) / np.sqrt(len(feats)))
    ys, xs = np.mgrid[0:gh, 0:gw]
    pos = np.stack([ys.ravel() / max(gh, gw), xs.ravel() / max(gh, gw)], axis=1) * position_weight
    return kmeans(np.hstack(chunks + [pos]), k).numpy().reshape(gh, gw) + 1


def vote(score, parts):
    """score [H, W] (> 0 person 1), parts [H, W] int (0 = no part) -> label [H, W] (0 person 1, 1 person 2)."""
    import numpy as np
    lab = (score <= 0).astype(np.uint8)
    ids = parts.ravel()
    sums = np.bincount(ids, weights=score.ravel(), minlength=int(ids.max()) + 1)
    out = (sums <= 0).astype(np.uint8)[ids].reshape(parts.shape)
    out[parts == 0] = lab[parts == 0]
    return out


# ---------------------------------------------------------------- GPU path (torch on the sampling device)

def kmeans_grid(X, k, gh, gw, iters=25):
    """k-means for tokens on a gh x gw grid, GPU friendly: the first centres are tokens spread evenly over the
    grid (no sequential k-means++ draws) and a fixed number of rounds (no convergence check, so no host sync)."""
    ny = max(1, round(math.sqrt(k * gh / gw)))
    nx = max(1, math.ceil(k / ny))
    ys = ((torch.arange(ny, device=X.device) + 0.5) * gh / ny).long().clamp(max=gh - 1)
    xs = ((torch.arange(nx, device=X.device) + 0.5) * gw / nx).long().clamp(max=gw - 1)
    idx = (ys[:, None] * gw + xs[None, :]).flatten()[:k]
    c = X[idx].clone()
    k = len(idx)
    for _ in range(iters):
        lab = torch.cdist(X, c).argmin(1)
        cnt = torch.bincount(lab, minlength=k)
        new = torch.zeros_like(c).index_add_(0, lab, X) / cnt.clamp_min(1)[:, None].float()
        c = torch.where((cnt > 0)[:, None], new, c)
    return torch.cdist(X, c).argmin(1)


def token_parts_t(feats, gh, gw, k, position_weight, device):
    """token_parts on `device`: list of [S, D] features -> part ids [gh, gw] (1..k), long."""
    chunks = []
    for x in feats:
        x = x.to(device).float()
        chunks.append(x / x.norm(dim=1, keepdim=True).clamp_min(1e-6) / math.sqrt(len(feats)))
    ys, xs = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
    pos = torch.stack([ys.flatten(), xs.flatten()], 1).float() / max(gh, gw) * position_weight
    return kmeans_grid(torch.cat(chunks + [pos], 1), k, gh, gw).view(gh, gw) + 1


def vote_t(score, parts):
    """vote on tensors: score [H, W] float (> 0 person 1), parts [H, W] long -> label [H, W] uint8."""
    ids = parts.flatten()
    sums = torch.zeros(int(ids.max()) + 1, device=score.device).scatter_add_(0, ids, score.flatten().float())
    out = (sums <= 0).to(torch.uint8)[ids].view(parts.shape)
    return torch.where(parts == 0, (score <= 0).to(torch.uint8), out)


def rgb_to_lab(rgb):
    """[H, W, 3] sRGB 0..1 -> CIE Lab (D65), as skimage.color.rgb2lab."""
    c = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]], device=rgb.device)
    xyz = c @ m.T / torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def slic_t(rgb, n_segments=800, compactness=10.0, iters=10):
    """SLIC superpixels on the GPU: [H, W, 3] 0..1 -> labels [H, W] long (1..). Same distance as skimage
    (Lab colour, spatial term scaled by compactness / grid step); each pixel is compared with the centres of the
    3x3 grid cells around its own cell. No connectivity enforcement (small stray pieces can stay)."""
    H, W = rgb.shape[:2]
    dev = rgb.device
    step = math.sqrt(H * W / n_segments)
    ny, nx = max(1, round(H / step)), max(1, round(W / step))
    sy, sx = H / ny, W / nx
    lab = rgb_to_lab(rgb.float())
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    feat = torch.cat([lab, ys[..., None], xs[..., None]], -1).view(-1, 5)          # [N, 5]
    cy = ((torch.arange(ny, device=dev) + 0.5) * sy).view(-1, 1).expand(ny, nx)
    cx = ((torch.arange(nx, device=dev) + 0.5) * sx).view(1, -1).expand(ny, nx)
    ci = cy.long().clamp(max=H - 1) * W + cx.long().clamp(max=W - 1)
    centres = torch.cat([lab.reshape(-1, 3)[ci.flatten()], cy.reshape(-1, 1), cx.reshape(-1, 1)], 1)   # [K, 5]
    cell_y = (ys / sy).long().clamp(max=ny - 1).flatten()
    cell_x = (xs / sx).long().clamp(max=nx - 1).flatten()
    cand = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = cell_y + dy, cell_x + dx
            ok = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            cand.append(torch.where(ok, yy.clamp(0, ny - 1) * nx + xx.clamp(0, nx - 1), -1))
    cand = torch.stack(cand)                                                       # [9, N]
    valid = cand >= 0
    cand0 = cand.clamp_min(0)
    spatial = (compactness / step) ** 2
    K = ny * nx
    for _ in range(iters):
        c = centres[cand0]                                                         # [9, N, 5]
        d = ((c[..., :3] - feat[None, :, :3]) ** 2).sum(-1) + spatial * ((c[..., 3:] - feat[None, :, 3:]) ** 2).sum(-1)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        ids = cand0.gather(0, d.argmin(0, keepdim=True))[0]
        sums = torch.zeros(K, 5, device=dev).index_add_(0, ids, feat)
        cnt = torch.bincount(ids, minlength=K).float()
        keep = cnt > 0
        centres[keep] = sums[keep] / cnt[keep, None]
    return ids.view(H, W) + 1


def auto_device():
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_windows(k, mode):
    tail = (k - max(1, k // 3), k)
    return {"all": [(0, k)], "tail": [tail], "both": [(0, k), tail]}[mode]


def auto_label(rec, image, weights, backend="gpu"):
    """Pass-1 recorder + pass-1 image [H, W, 3] (0..1) -> (label [H, W] uint8, info).
    backend "gpu": torch on the sampling device (SLIC included); "cpu": numpy / skimage (the fitted reference)."""
    if backend == "gpu":
        return _auto_label_t(rec, image, weights, auto_device())
    import numpy as np
    from PIL import Image
    from skimage.segmentation import slic
    order = sorted(rec.steps)
    k = len(order)
    T, gh, gw = rec.grid
    H, W = image.shape[:2]
    table = {int(key): v for key, v in weights["probe_steps"].items()}
    key = pick_k(table, k)
    entry = table[key]
    mode = entry.get("loso", {}).get("window") or ("both" if len(entry["windows"]) > 1 else "all")
    wins = auto_windows(k, mode)
    nb = 1 + max(b for _, b in rec.blocks)
    if nb != weights["blocks"]:
        raise RuntimeError(f"[SnapRefRegionAutoMask] weights are for {weights['blocks']} blocks, recorded {nb}")
    feats = torch.cat([block_features(rec, order[a:b], T, gh, gw) for a, b in wins])
    w = torch.tensor(entry["weights"], dtype=torch.float32)
    score = weighted_score(feats, w)
    score_img = upsample(score[None], H, W)[0].numpy().astype(np.float64)
    recorded = sorted({i for i, _ in rec.features if i < k})
    if not recorded:
        raise RuntimeError("[SnapRefRegionAutoMask] no token features recorded; set auto_mask on the probe")
    f_step = entry["feature_step"] if entry["feature_step"] in recorded else recorded[-1]
    parts = token_parts([rec.features[(f_step, b)] for b in weights["feature_blocks"]], gh, gw,
                        int(weights["parts"]), float(weights["position_weight"]))
    parts = np.array(Image.fromarray(parts.astype(np.int32)).resize((W, H), Image.NEAREST)).astype(np.int64)
    coarse = vote(score_img, parts)
    rgb = (image.clamp(0, 1).numpy() * 255).astype(np.uint8)
    sp = slic(rgb, n_segments=int(weights["slic_segments"]), compactness=10, start_label=1).astype(np.int64)
    label_img = vote(np.where(coarse == 0, 1.0, -1.0), sp)
    info = {"probe_steps": k, "weights_k": key, "windows": [list(x) for x in wins], "feature_step": f_step,
            "area_1": round(float((label_img == 0).mean()), 4), "area_2": round(float((label_img == 1).mean()), 4)}
    return label_img, info


class SnapRefRegionAutoMask:
    """Pass-1 image + probe -> mask_1 / mask_2 for SnapRef Region Mask (two-stage generation)."""

    @classmethod
    def INPUT_TYPES(cls):
        names = auto_weight_names() or ["none"]
        return {"required": {
            "image": ("IMAGE", {"tooltip": "the decoded image the probe recorded (a short pass 1, or the x0 of a partial SnapRef pass)"}),
            "probe": ("REGION_PROBE", {"tooltip": "the probe that wrapped the pass-1 sampler (auto_mask set)"}),
            "weights": (names, {"default": names[0]}),
        }}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2")
    FUNCTION = "apply"
    # runs even when nothing consumes the masks (estimation only without a panel): the dump is its output then
    OUTPUT_NODE = True
    CATEGORY = "SnapRef"
    DESCRIPTION = "Estimates where person 1 / person 2 are in the pass-1 image; every pixel goes to one of them."

    def apply(self, image, probe, weights):
        rec = probe["recorder"]
        if not rec.steps or rec.grid is None:
            raise RuntimeError("[SnapRefRegionAutoMask] nothing recorded; is the probe the last model node of pass 1?")
        label_img, info = auto_label(rec, image[0].float().cpu(), load_auto_weights(weights))
        logging.info("[SnapRefRegionAutoMask] %s", info)
        lab = torch.from_numpy(label_img)
        if probe.get("dump_path"):
            rec.dump(probe["dump_path"])
            from PIL import Image
            Image.fromarray(label_img).save(os.path.splitext(probe["dump_path"])[0] + "_label.png")
        return ((lab == 0).float()[None], (lab == 1).float()[None])


class SnapRefRegionAutoMaskPanel:
    """Estimation image and final image, each with the estimated person regions over it, plus a caption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "probe_image": ("IMAGE",), "final_image": ("IMAGE",), "mask_1": ("MASK",), "mask_2": ("MASK",),
        }, "optional": {"caption": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panel",)
    FUNCTION = "apply"
    CATEGORY = "SnapRef"

    def apply(self, probe_image, final_image, mask_1, mask_2, caption=""):
        a = probe_image[0].float().cpu()
        b = final_image[0].float().cpu()
        H, W = b.shape[:2]
        if a.shape[:2] != (H, W):
            a = F.interpolate(a.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        masks = torch.stack([mask_1[0], mask_2[0]]).float().cpu() > 0.5
        if masks.shape[1:] != (H, W):
            masks = F.interpolate(masks[None].float(), size=(H, W), mode="nearest")[0] > 0.5
        row = torch.cat([label(a, "estimation image"), label(assign_panel(a, masks), "estimated regions"),
                         label(assign_panel(b, masks), "final (SnapRef) + regions"), label(b, "final")], dim=1)
        legend = "red = person 1 (SnapRef image)   blue = person 2 (SnapRef image_2)"
        bar = caption_bar(row.shape[1], (caption.strip() + "\n" if caption.strip() else "") + legend, H)
        panel = torch.cat([bar, row], dim=0) if bar is not None else row
        return (panel[None],)


def _auto_parts(rec, weights):
    order = sorted(rec.steps)
    k = len(order)
    table = {int(key): v for key, v in weights["probe_steps"].items()}
    key = pick_k(table, k)
    entry = table[key]
    mode = entry.get("loso", {}).get("window") or ("both" if len(entry["windows"]) > 1 else "all")
    wins = auto_windows(k, mode)
    nb = 1 + max(b for _, b in rec.blocks)
    if nb != weights["blocks"]:
        raise RuntimeError(f"[SnapRefRegionAutoMask] weights are for {weights['blocks']} blocks, recorded {nb}")
    recorded = sorted({i for i, _ in rec.features if i < k})
    if not recorded:
        raise RuntimeError("[SnapRefRegionAutoMask] no token features recorded; set auto_mask on the probe")
    f_step = entry["feature_step"] if entry["feature_step"] in recorded else recorded[-1]
    return order, k, key, entry, wins, f_step


def _auto_label_t(rec, image, weights, device):
    order, k, key, entry, wins, f_step = _auto_parts(rec, weights)
    T, gh, gw = rec.grid
    H, W = image.shape[:2]
    feats = torch.cat([block_features(rec, order[a:b], T, gh, gw, device) for a, b in wins])
    score = weighted_score(feats, torch.tensor(entry["weights"], dtype=torch.float32, device=device))
    score_img = upsample(score[None], H, W)[0]
    parts = token_parts_t([rec.features[(f_step, b)] for b in weights["feature_blocks"]], gh, gw,
                          int(weights["parts"]), float(weights["position_weight"]), device)
    parts = F.interpolate(parts[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
    coarse = vote_t(score_img, parts)
    sp = slic_t(image.to(device).float().clamp(0, 1), n_segments=int(weights["slic_segments"]), compactness=10.0)
    label_img = vote_t(torch.where(coarse == 0, 1.0, -1.0), sp).cpu().numpy()
    info = {"probe_steps": k, "weights_k": key, "windows": [list(x) for x in wins], "feature_step": f_step,
            "backend": f"gpu ({device})",
            "area_1": round(float((label_img == 0).mean()), 4), "area_2": round(float((label_img == 1).mean()), 4)}
    return label_img, info


def subsample_sigmas(sigmas, steps):
    """Keep `steps` intervals of a sigma schedule: first and last value kept, evenly spaced in between."""
    n = len(sigmas) - 1
    if steps >= n:
        return sigmas
    idx = sorted({round(i * n / steps) for i in range(steps + 1)})
    return sigmas[idx]


class SnapRefSubsampleSigmas:
    """Finish a schedule in fewer steps: e.g. the low part of SplitSigmas (steps k..30) in 4 steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("SIGMAS",),
                             "steps": ("INT", {"default": 4, "min": 1, "max": 1000})}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "apply"
    CATEGORY = "SnapRef"

    def apply(self, sigmas, steps):
        return (subsample_sigmas(sigmas, steps),)


NODE_CLASS_MAPPINGS = {"SnapRefRegionProbe": SnapRefRegionProbe,
                       "SnapRefRegionProbeOverlay": SnapRefRegionProbeOverlay,
                       "SnapRefSubsampleSigmas": SnapRefSubsampleSigmas,
                       "SnapRefRegionAutoMask": SnapRefRegionAutoMask,
                       "SnapRefRegionAutoMaskPanel": SnapRefRegionAutoMaskPanel}
NODE_DISPLAY_NAME_MAPPINGS = {"SnapRefRegionProbe": "SnapRef Region Probe",
                              "SnapRefRegionProbeOverlay": "SnapRef Region Probe Overlay",
                              "SnapRefRegionAutoMask": "SnapRef Region Auto Mask",
                              "SnapRefRegionAutoMaskPanel": "SnapRef Region Auto Mask Panel",
                              "SnapRefSubsampleSigmas": "SnapRef Subsample Sigmas"}
