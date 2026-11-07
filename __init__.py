"""ComfyUI-SnapRef-RegionMask: limit where each SnapRef reference is read.

SnapRef (https://github.com/ArcAnima/SnapRef) prepends the K/V of a reference image to every
self-attention call of Anima, so every image token can copy the reference. With two different
characters (SnapRef `image` + `image_2`) both references therefore leak into both people.

This node adds a per-token log bias to the reference keys of each reference: 0 inside that
reference's box, log(outside_strength) outside it (-inf when 0). Tokens outside the box get plain
self-attention for that reference. For the first `ref_start` fraction of the steps no token reads
any reference, so the layout of the scene is decided by the prompt first.

Place it right after SnapRef. It works alone or together with Anima Regional prompting
(which patches cross-attention only). Masking runs on SnapRef's plain path (fidelity 1,
ref_guidance off, ref_in_uncond off, CUDA fp16/bf16); on other paths SnapRef runs unchanged.
"""
import logging
import math

import torch
import comfy.patcher_extension

__version__ = "0.4.0"

KEY = "snapref_region_mask"
OUTER_KEY = "snapref_region_mask_outer"
FULL_BATCH_KEY = "snapref_region_mask_full_batch"
SNAPREF_KEY = "snapref"   # SnapRef's wrapper key (snapref.WRAPPER_KEY)
_REGION = {"bias": None}   # per-call token biases, one tensor per reference box
# transformer_options key of a callback(block_index, [lse per reference]) set by SnapRef Region Probe (ref_affinity):
# how strongly each image token attends to each reference before the region bias
AFFINITY_KEY = "snapref_ref_affinity"


def _lse_attention(q, k, v):
    """Fused attention with its log-sum-exp (same kernels as SnapRef).

    q, k, v [B, H, S, D] -> out float32 [B, H, S, D], lse float32 [B, H, S, 1]."""
    try:
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)[:2]
    except (RuntimeError, AttributeError):
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)[:2]
    return out.float(), lse.float()[..., :q.shape[2], None]


def _attach(dm):
    """Wrap SnapRef's self-attention forward on every block for one model call.

    Returns the (module, previous forward) pairs; the caller restores them in a finally block, so the
    model is left exactly as SnapRef set it up whether or not this node is in the graph."""
    restore = []
    for index, block in enumerate(dm.blocks):
        attn = block.self_attn
        if getattr(attn, "_ref_kv_state", "missing") == "missing":
            continue  # SnapRef has not installed its hook on this block
        snapref_forward = attn.forward

        def forward(x, context=None, rope_emb=None, transformer_options={}, _attn=attn, _prev=snapref_forward,
                    _index=index):
            state = _attn._ref_kv_state
            bias_tok = _REGION["bias"]
            if (state is None or bias_tok is None or "record" in state or state.get("rows") is not None
                    or not (x.is_cuda and x.dtype in (torch.float16, torch.bfloat16))):
                return _prev(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            q, k, v = _attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            B, S, H, D = q.shape
            if bias_tok[0].shape[0] != S:
                logging.warning("[SnapRefRegionMask] token count %d != mask %d; mask skipped", S, bias_tok[0].shape[0])
                return _prev(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            k_r, v_r = state["cache"][_index]
            sizes, biases = state["sizes"], state["biases"]
            qh = q.transpose(1, 2)
            # softmax over [target keys; reference keys + bias] computed group by group and merged by log-sum-exp
            parts = [(0.0, *_lse_attention(qh, k.transpose(1, 2), v.transpose(1, 2)))]
            start = 0
            lses = []
            for gi, (size, b) in enumerate(zip(sizes, biases)):
                # reference gi reads box gi; the last box repeats for extra references
                tok = bias_tok[min(gi, len(bias_tok) - 1)].to(q.device).view(1, 1, S, 1)
                kr = k_r[:, start:start + size].expand(B, -1, -1, -1).to(q.dtype).transpose(1, 2)
                vr = v_r[:, start:start + size].expand(B, -1, -1, -1).to(q.dtype).transpose(1, 2)
                start += size
                out_r, lse_r = _lse_attention(qh, kr, vr)
                lses.append(lse_r)
                parts.append((tok + b, out_r, lse_r))
            record = transformer_options.get(AFFINITY_KEY)
            if record is not None:
                record(_index, lses)
            m = torch.stack([(lse + b).expand_as(parts[0][2]) for b, _, lse in parts]).amax(dim=0)
            num, den = 0.0, 0.0
            for b, o, lse in parts:
                w = torch.exp(lse + b - m)
                num, den = num + w * o, den + w
            out = (num / den).to(q.dtype).transpose(1, 2).reshape(B, S, H * D)
            return _attn.output_dropout(_attn.output_proj(out))

        restore.append((attn, snapref_forward))
        attn.forward = forward
    return restore


def _detach(restore):
    for attn, previous in reversed(restore):
        attn.forward = previous


def box_mask(gh, gw, box, feather, outside, device="cpu"):
    """Log-bias grid [gh, gw] for one box (x, y, width, height as fractions of the image).

    Inside the box 0, outside log(outside) (-inf when outside is 0), with a linear ramp of
    `feather` (fraction of the image) centred on the box edges."""
    x0, y0, bw, bh = box
    ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) / gh
    xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) / gw

    def ramp(t, lo, hi):
        if feather <= 0:
            return ((t >= lo) & (t <= hi)).float()
        # no soft edge on a side that lies on the image border
        rise = torch.ones_like(t) if lo <= 0 else (t - lo) / feather + 0.5
        fall = torch.ones_like(t) if hi >= 1 else (hi - t) / feather + 0.5
        return torch.clamp(torch.minimum(rise, fall), 0.0, 1.0)

    m = ramp(ys, y0, y0 + bh)[:, None] * ramp(xs, x0, x0 + bw)[None, :]
    m = outside + (1.0 - outside) * m
    return torch.log(m.clamp_min(1e-30)).masked_fill(m <= 0, float("-inf"))


def mask_grid(mask, gh, gw, outside, device="cpu"):
    """Log-bias grid [gh, gw] from a ComfyUI MASK ([B, H, W] or [H, W], 1 = read the reference).

    The first mask of a batch is used. It is resized to the token grid by area averaging, so a soft
    or anti-aliased edge becomes a soft edge here too."""
    m = mask[0] if mask.ndim == 3 else mask
    m = m.to(device=device, dtype=torch.float32)[None, None]
    m = torch.nn.functional.interpolate(m, size=(gh, gw), mode="area")[0, 0].clamp(0.0, 1.0)
    m = outside + (1.0 - outside) * m
    return torch.log(m.clamp_min(1e-30)).masked_fill(m <= 0, float("-inf"))


def region_bias(region, gh, gw, feather, outside, device):
    """region = ("box", (x, y, w, h)) or ("mask", MASK tensor) -> flattened log-bias [gh * gw]."""
    kind, value = region
    if kind == "mask":
        return mask_grid(value, gh, gw, outside, device).flatten()
    return box_mask(gh, gw, value, feather, outside, device).flatten()


def sampling_progress(options):
    """Fraction of the sampling schedule already done (0 at the first step), or None if unknown."""
    sig, sched = options.get("sigmas"), options.get("sample_sigmas")
    if sig is None or sched is None or len(sched) < 2:
        return None
    s = float(sig.flatten()[0])
    sched = [float(v) for v in sched.flatten()]
    idx = next((i for i, v in enumerate(sched) if v <= s + 1e-6), len(sched) - 1)
    return idx / (len(sched) - 1)


def _outer_wrapper(executor, x, timesteps, context, *args, **kwargs):
    """Runs outside SnapRef: records the full batch size so the inner wrapper can spot SnapRef's split calls."""
    opts = kwargs.get("transformer_options", {})
    kwargs["transformer_options"] = {**opts, FULL_BATCH_KEY: int(x.shape[0])}
    return executor(x, timesteps, context, *args, **kwargs)


def _fix_split_markers(dm, x, kwargs):
    """SnapRef runs a mixed cond/uncond batch as two calls but passes the original cond_or_uncond list
    to both. Code that reads the markers per call (e.g. Anima Regional) then treats the negative half as
    conditional. Inside a split call, rewrite the markers to the one half actually being run:
    SnapRef reads the reference only in the cond half (ref_in_uncond off is the only case that splits)."""
    opts = kwargs.get("transformer_options", {})
    flags, full = opts.get("cond_or_uncond"), opts.get(FULL_BATCH_KEY)
    if not flags or full is None or int(x.shape[0]) == full or len(set(flags)) < 2:
        return
    with_ref = getattr(dm.blocks[0].self_attn, "_ref_kv_state", None) is not None
    kwargs["transformer_options"] = {**opts, "cond_or_uncond": [0 if with_ref else 1]}


def _make_wrapper(regions, feather, outside, ref_start):
    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        dm = executor.class_obj
        _fix_split_markers(dm, x, kwargs)
        p = int(getattr(dm, "patch_spatial", 2))
        T = x.shape[2] if x.ndim == 5 else 1
        gh, gw = math.ceil(x.shape[-2] / p), math.ceil(x.shape[-1] / p)
        progress = sampling_progress(kwargs.get("transformer_options", {}))
        if ref_start > 0 and progress is not None and progress < ref_start:
            masks = [torch.full((gh * gw,), float("-inf"), device=x.device)]  # layout phase: no reference
        else:
            masks = [region_bias(r, gh, gw, feather, outside, x.device) for r in regions]
        _REGION["bias"] = [m.repeat(T) for m in masks]
        restore = _attach(dm)   # SnapRef (outer wrapper) has already installed its hook for this call
        try:
            return executor(x, timesteps, context, *args, **kwargs)
        finally:
            _detach(restore)
            _REGION["bias"] = None
    return wrapper


LAYOUTS = ["left_right", "top_bottom", "boxes"]


def layout_boxes(layout, split, box_1, box_2):
    """Boxes (x, y, w, h) for references 1 and 2. boxes layout: box 2 only when its width is > 0."""
    split = min(max(float(split), 0.0), 1.0)
    if layout == "left_right":
        return [(0.0, 0.0, split, 1.0), (split, 0.0, 1.0 - split, 1.0)]
    if layout == "top_bottom":
        return [(0.0, 0.0, 1.0, split), (0.0, split, 1.0, 1.0 - split)]
    if layout != "boxes":
        raise ValueError(f"[SnapRefRegionMask] unknown layout {layout!r}")
    return [box_1] + ([box_2] if box_2[2] > 0 else [])


def _frac(default, tip):
    return ("FLOAT", {"default": default, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tip})


class SnapRefRegionMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "the MODEL output of SnapRef"}),
            "layout": (LAYOUTS, {"default": "left_right",
                                 "tooltip": "left_right: reference 1 (image) left of `split`, reference 2 (image_2) "
                                            "right. top_bottom: reference 1 above `split`, reference 2 below. "
                                            "boxes: use x / y / width / height and x_2 ... height_2. "
                                            "A connected mask_1 / mask_2 overrides the region of that reference"}),
            "split": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01,
                                "tooltip": "left_right / top_bottom: where the two regions meet, as a fraction of "
                                           "the image width / height"}),
            "feather": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                                  "tooltip": "soft edge width around each region, fraction of the image"}),
            "outside_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "how much a reference is still read outside its box (0 = not at all)"}),
            "ref_start": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.95, "step": 0.05,
                                    "tooltip": "fraction of the steps with no reference at all, so the prompt decides "
                                               "the layout first. 0.1-0.3; higher keeps the layout, lower keeps more "
                                               "of the references"}),
        }, "optional": {
            "x": _frac(0.0, "layout boxes: box of SnapRef `image` (reference 1), left edge, fraction of the width"),
            "y": _frac(0.0, "layout boxes: box of reference 1, top edge, fraction of the height"),
            "width": _frac(0.5, "layout boxes: box of reference 1, width"),
            "height": _frac(1.0, "layout boxes: box of reference 1, height"),
            "x_2": _frac(0.5, "layout boxes: box of SnapRef `image_2` (reference 2), left edge"),
            "y_2": _frac(0.0, "layout boxes: box of reference 2, top edge"),
            "width_2": _frac(0.0, "layout boxes: box of reference 2, width. 0 = reference 2 (and 3) use box 1"),
            "height_2": _frac(1.0, "layout boxes: box of reference 2, height"),
            "mask_1": ("MASK", {"tooltip": "region of reference 1 as a MASK (white = reads the reference). "
                                         "Overrides the box x / y / width / height"}),
            "mask_2": ("MASK", {"tooltip": "region of reference 2 as a MASK. Overrides the box x_2 ... height_2"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "SnapRef"
    DESCRIPTION = ("Limits where each SnapRef reference is read: reference 1 inside region 1, reference 2 inside "
                   "region 2. A region is a box (fractions of the image) or a MASK. Use SnapRef multi_layout = "
                   "separate with one character per reference.")

    def apply(self, model, layout="left_right", split=0.5, feather=0.05, outside_strength=0.0, ref_start=0.1,
              x=0.0, y=0.0, width=0.5, height=1.0, x_2=0.5, y_2=0.0, width_2=0.0, height_2=1.0,
              mask_1=None, mask_2=None):
        boxes = layout_boxes(layout, split, (x, y, width, height), (x_2, y_2, width_2, height_2))
        regions = [("mask", mask_1) if mask_1 is not None else ("box", boxes[0])]
        if mask_2 is not None:
            regions.append(("mask", mask_2))
        elif len(boxes) > 1:
            regions.append(("box", boxes[1]))
        m = model.clone()
        dmw = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
        m.add_wrapper_with_key(dmw, KEY, _make_wrapper(regions, feather, outside_strength, ref_start))
        m.add_wrapper_with_key(dmw, OUTER_KEY, _outer_wrapper)
        wrappers = m.wrappers.get(dmw, {})
        if SNAPREF_KEY not in wrappers:
            logging.warning("[SnapRefRegionMask] no SnapRef on this model; connect it right after SnapRef")
        else:
            # order: batch recorder, SnapRef (installs its attention hook and splits cond/uncond), then this node
            outer = wrappers.pop(OUTER_KEY)
            snapref = wrappers.pop(SNAPREF_KEY)
            m.wrappers[dmw] = {OUTER_KEY: outer, SNAPREF_KEY: snapref, **wrappers}
        return (m,)


def _load_probe():
    """region_probe.py as a package submodule inside ComfyUI, or by path when this file is loaded alone (tests)."""
    try:
        from . import region_probe
        return region_probe
    except ImportError:
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "snapref_region_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "region_probe.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_probe = _load_probe()
NODE_CLASS_MAPPINGS = {"SnapRefRegionMask": SnapRefRegionMask, **_probe.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {"SnapRefRegionMask": "SnapRef Region Mask", **_probe.NODE_DISPLAY_NAME_MAPPINGS}
