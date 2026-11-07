"""Unit tests that run without ComfyUI: python -m pytest tests"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import pytest

comfy = types.ModuleType("comfy")
pe = types.ModuleType("comfy.patcher_extension")
pe.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
comfy.patcher_extension = pe
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.patcher_extension", pe)

spec = importlib.util.spec_from_file_location("snapref_region_probe", Path(__file__).resolve().parents[1] / "region_probe.py")
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


def test_term_positions_variants_and_end_token():
    prompt = [5, 7, 9, 5, 7, 3, 1]
    assert node.term_positions(prompt, [[5, 7, 1]]) == [0, 1, 3, 4]
    assert node.term_positions(prompt, [[9, 1], [3, 1]]) == [2, 5]
    assert node.term_positions(prompt, [[8, 1]]) == []
    assert node.split_terms(" 1girl | girl |") == ["1girl", "girl"]


def test_cond_rows():
    assert node.cond_rows([0, 1], 2) == [0]
    assert node.cond_rows([1, 0], 4) == [2, 3]
    assert node.cond_rows([0, 1], 1) is None   # split call
    assert node.cond_rows(None, 2) is None


def test_cross_attention_share_matches_manual_softmax():
    torch.manual_seed(0)
    q, k = torch.randn(1, 6, 2, 4), torch.randn(1, 5, 2, 4)
    share = node.cross_attention_share(q, k, [[0, 1], [3]])
    probs = (torch.einsum("bshd,bnhd->bhsn", q, k) / 2.0).softmax(-1)[0]   # D=4 -> scale 1/2
    assert torch.allclose(share[0, 0], probs[..., [0, 1]].sum(-1).mean(0), atol=1e-6)
    assert torch.allclose(share[1, 0], probs[..., 3].mean(0), atol=1e-6)


def test_assign_foreground_and_argmax():
    norm = torch.tensor([[[1.0, 0.2, 0.1]], [[0.5, 0.9, 0.1]]])
    masks, used = node.assign(norm, 0.3)
    assert used == 0.3
    assert masks[0].tolist() == [[True, False, False]]
    assert masks[1].tolist() == [[False, True, False]]


def test_otsu_splits_bimodal_values():
    values = torch.cat([torch.full((100,), 0.1), torch.full((100,), 0.8)])
    t = node.otsu(values)
    assert 0.1 < t < 0.8
    masks, used = node.assign(torch.stack([values.view(1, -1), values.view(1, -1) * 0.5]), 0.0)
    assert used == t and masks[0].sum() == 100


def test_sampling_progress_nearest_step():
    sched = torch.tensor([1.0, 0.5, 0.25, 0.0])
    assert node.sampling_progress({"sigmas": torch.tensor([0.9999]), "sample_sigmas": sched}) == (0, 0.0)
    assert node.sampling_progress({"sigmas": torch.tensor([0.25]), "sample_sigmas": sched}) == (2, 2 / 3)


def test_padding_keys_are_excluded():
    torch.manual_seed(0)
    q, k = torch.randn(1, 6, 2, 4), torch.randn(1, 5, 2, 4)
    k[:, 3:] = 0
    share = node.cross_attention_share(q, k, [[0]], n_keys=3)
    probs = (torch.einsum("bshd,bnhd->bhsn", q, k[:, :3]) / 2.0).softmax(-1)[0]
    assert torch.allclose(share[0, 0], probs[..., 0].mean(0), atol=1e-6)


def test_recorder_windows():
    rec = node.Recorder(["a", "b"], [[0], [1]])
    for idx, value in ((0, 1.0), (1, 3.0), (2, 5.0)):
        rec.progress[idx] = idx / 10
        rec.add(idx, torch.full((2, 4), value))
        rec.add(idx, torch.full((2, 4), value))   # two blocks
    mean, picked = rec.window(0.15)
    assert picked == [0, 1] and torch.allclose(mean, torch.full((2, 4), 2.0))
    mean, picked = rec.window(1.0)
    assert picked == [0, 1, 2] and torch.allclose(mean, torch.full((2, 4), 3.0))


class FakeAttn:
    def __init__(self, heads=2, dim=4):
        self.heads, self.dim = heads, dim
        self.calls = 0

    def compute_qkv(self, x, context, rope_emb=None, transformer_options={}):
        B, S, _ = x.shape
        q = x[..., :self.heads * self.dim].reshape(B, S, self.heads, self.dim)
        k = context[..., :self.heads * self.dim].reshape(B, context.shape[1], self.heads, self.dim)
        return q, k, k

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        self.calls += 1
        return x * 2


class FakeDM:
    patch_spatial = 2

    def __init__(self):
        self.blocks = [types.SimpleNamespace(cross_attn=FakeAttn()) for _ in range(2)]


class FakeExecutor:
    def __init__(self, dm, seq):
        self.class_obj = dm
        self.seq = seq

    def __call__(self, x, timesteps, context, *args, transformer_options={}, **kwargs):
        # SnapRef-like split: run cond row and uncond row as separate calls
        outs = []
        for r in range(x.shape[0]):
            h = torch.randn(1, self.seq, 8)
            for block in self.class_obj.blocks:
                h = block.cross_attn.forward(h, context[r:r + 1], transformer_options=transformer_options)
            outs.append(h)
        return torch.cat(outs)


def test_wrapper_records_cond_rows_only_and_restores_forward():
    torch.manual_seed(0)
    dm = FakeDM()
    originals = [b.cross_attn.forward for b in dm.blocks]
    rec = node.Recorder(["a", "b"], [[0], [1, 2]])
    wrapper = node._make_wrapper(rec)
    x = torch.zeros(2, 16, 1, 4, 4)            # latent 4x4 -> grid 2x2 = 4 tokens
    context = torch.randn(2, 6, 8)             # row 0 cond, row 1 uncond
    sched = torch.tensor([3.0, 2.0, 1.0, 0.0])
    for step in range(3):
        opts = {"cond_or_uncond": [0, 1], "sigmas": sched[step:step + 1], "sample_sigmas": sched}
        wrapper(FakeExecutor(dm, 4), x, None, context, transformer_options=opts)
    assert [b.cross_attn.forward for b in dm.blocks] == originals
    assert sorted(rec.steps) == [0, 1, 2]
    assert all(rec.counts[i] == 2 for i in rec.steps)       # 2 blocks x 1 cond row
    assert rec.grid == (1, 2, 2) and rec.total_steps == 3
    assert rec.steps[0].shape == (2, 4)
    # a new sampling run resets the recorder
    opts = {"cond_or_uncond": [0, 1], "sigmas": sched[0:1], "sample_sigmas": sched}
    wrapper(FakeExecutor(dm, 4), x, None, context, transformer_options=opts)
    assert sorted(rec.steps) == [0]


def test_segments_record_under_whole_schedule_index():
    torch.manual_seed(0)
    dm = FakeDM()
    rec = node.Recorder(["a", "b"], [[0], [1, 2]])
    full = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])           # 5 steps, split after step 2
    x = torch.zeros(2, 16, 1, 4, 4)
    context = torch.randn(2, 6, 8)
    for sched, offset in ((full[:3], 0), (full[2:], 2)):
        wrapper = node._make_wrapper(rec, offset, 5)
        for i in range(len(sched) - 1):
            opts = {"cond_or_uncond": [0, 1], "sigmas": sched[i:i + 1], "sample_sigmas": sched}
            wrapper(FakeExecutor(dm, 4), x, None, context, transformer_options=opts)
    assert sorted(rec.steps) == [0, 1, 2, 3, 4]                  # second segment did not reset the first
    assert rec.progress[3] == 3 / 5 and rec.total_steps == 5


def test_overlay_outputs_panel_masks_and_info():
    rec = node.Recorder(["1girl", "1boy"], [[0], [1]])
    rec.grid, rec.total_steps = (1, 2, 2), 10
    left = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    for idx in range(4):
        rec.progress[idx] = idx / 10
        rec.add(idx, left + 0.01 * idx)
    image = torch.rand(1, 16, 16, 3)
    out = node.SnapRefRegionProbeOverlay().apply(image, {"recorder": rec, "prompt_tokens": 9}, window_end=0.2,
                                                 caption="seed 1\nprompt: 1girl, 1man")
    panel, m1, m2 = out["result"]
    # caption bar + summary row + two rows of first-k-steps splits
    assert panel.shape[0] == 1 and panel.shape[2:] == (64, 3) and panel.shape[1] > 48
    assert torch.allclose(panel[0, :8, :8], torch.ones(8, 8, 3))   # white caption bar on top
    assert m1.shape == (1, 16, 16) and m2.shape == (1, 16, 16)
    assert m1[0, :, :4].all() and m2[0, :, -4:].all()   # person 1 left, person 2 right
    assert not (m1.bool() & m2.bool()).any()
    info = json.loads(out["ui"]["snapref_probe_info"][0])
    assert info["windows"]["0.2"]["steps"] == [0, 1]
    assert set(info["windows"]) >= {"0.1", "0.2", "0.3", "1"}
    curve = info["step_curve"]["points"]
    assert [p["k"] for p in curve] == [1, 2, 3, 4]
    assert all(p["accuracy"] == 1.0 for p in curve)


def test_dump_saves_per_block_maps(tmp_dir=None):
    import tempfile
    import numpy as np
    rec = node.Recorder(["a", "b"], [[0], [1]], keep_blocks=True)
    rec.grid = (1, 1, 3)
    for idx in (0, 1):
        rec.progress[idx] = idx / 4
        for b in range(3):
            rec.add(idx, torch.full((2, 3), float(10 * idx + b)), block=b)
    with tempfile.TemporaryDirectory() as d:
        path = d + "/x/probe.npz"
        rec.dump(path)
        with np.load(path) as data:
            assert data["per_block"].shape == (2, 3, 2, 3)
            assert data["per_block"][1, 2, 0, 0] == 12
            assert list(data["steps"]) == [0, 1] and list(data["terms"]) == ["a", "b"]
    assert torch.allclose(rec.mean([0]), torch.full((2, 3), 1.0))   # block mean of 0, 1, 2


def test_features_are_pca_projected_and_dumped():
    import tempfile
    import numpy as np
    torch.manual_seed(0)
    rec = node.Recorder(["a", "b"], [[0], [1]], keep_blocks=True, keep_features=True)
    rec.grid = (1, 10, 10)
    step, block = node.FEATURE_STEPS[0], node.FEATURE_BLOCKS[0]
    rec.progress[step] = 0.1
    rec.add(step, torch.rand(2, 100), block=block)
    rec.add_feature(step, block, torch.randn(100, 300))
    rec.add_feature(step, 1, torch.randn(100, 300))          # block not in FEATURE_BLOCKS: ignored
    assert list(rec.features) == [(step, block)]
    assert rec.features[(step, block)].shape == (100, node.FEATURE_DIM)
    with tempfile.TemporaryDirectory() as d:
        rec.dump(d + "/f.npz")
        with np.load(d + "/f.npz") as data:
            assert data["features"].shape == (1, 1, 100, node.FEATURE_DIM)
            assert list(data["feature_steps"]) == [step] and list(data["feature_blocks"]) == [block]


def test_pick_k_and_bundled_weights():
    assert node.pick_k({1: 0, 3: 0, 8: 0}, 5) == 3
    assert node.pick_k({1: 0, 3: 0, 8: 0}, 30) == 8
    assert node.pick_k({3: 0, 8: 0}, 1) == 3
    assert "hakushi_anima_gt_v1" in node.auto_weight_names()
    w = node.load_auto_weights("hakushi_anima_gt_v1")
    assert sorted(int(k) for k in w["probe_steps"]) == [8, 12, 16, 20, 30]
    assert all(len(e["weights"]) == len(e["windows"]) * w["blocks"] + 1 for e in w["probe_steps"].values())
    assert node.load_block_weights("none") is None


def test_weighted_split_can_flip_a_block():
    # block 0 says person 1 is left, block 1 (inverted) says right; a negative weight on block 1 agrees with block 0
    rec = node.Recorder(["a", "b"], [[0], [1]], keep_blocks=True)
    rec.grid = (1, 1, 4)
    left = torch.tensor([[1.0, 0.9, 0.1, 0.0], [0.0, 0.1, 0.9, 1.0]])
    rec.progress[0] = 0.0
    rec.add(0, left, block=0)
    rec.add(0, left.flip(0), block=1)
    feats = node.block_features(rec, [0], 1, 1, 4)
    masks = node.score_split(node.weighted_score(feats, torch.tensor([1.0, -1.0, 0.0])))
    assert masks[0, 0].tolist() == [True, True, False, False]


def test_overlay_uses_block_weights():
    rec = node.Recorder(["1girl", "1boy"], [[0], [1]], keep_blocks=True)
    rec.grid, rec.total_steps = (1, 2, 2), 10
    left = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    for idx in range(4):
        rec.progress[idx] = idx / 10
        rec.add(idx, left + 0.01 * idx, block=0)
        rec.add(idx, 0.3 * (left + 0.01 * idx).flip(0), block=1)   # weaker, inverted block
    weights = {1: torch.tensor([1.0, -1.0, 0.0])}
    out = node.SnapRefRegionProbeOverlay().apply(
        torch.rand(1, 16, 16, 3), {"recorder": rec, "prompt_tokens": 9, "weights": weights,
                                   "block_weights": "test"}, window_end=0.2)
    info = json.loads(out["ui"]["snapref_probe_info"][0])
    assert info["block_weights"] == "test" and info["step_curve"]["split"] == "test"
    assert all(p["accuracy"] == 1.0 for p in info["step_curve"]["points"])
    _, m1, m2 = out["result"]
    assert m1[0, :, :4].any() and not (m1.bool() & m2.bool()).any()


def test_subsample_sigmas_keeps_ends():
    s = torch.linspace(1.0, 0.0, 28)   # 27 intervals, e.g. steps 3..30
    out = node.subsample_sigmas(s, 4)
    assert len(out) == 5 and out[0] == s[0] and out[-1] == 0
    assert torch.all(out[1:] < out[:-1])
    assert node.subsample_sigmas(s, 40) is s


def test_agreement_counts_swapped_split_as_wrong():
    ref = torch.tensor([[[True, True, False, False]], [[False, False, True, True]]])
    same = node.agreement(ref.clone(), ref)
    assert same["accuracy"] == 1.0 and same["iou"] == [1.0, 1.0]
    swapped = node.agreement(ref.flip(0), ref)
    assert swapped["accuracy"] == 0.0
    half = node.agreement(torch.tensor([[[True, True, True, False]], [[False, False, False, True]]]), ref)
    assert half["accuracy"] == 0.75


def test_weight_lists_are_split_by_kind():
    assert "hakushi_anima_gt_v1" in node.auto_weight_names()
    assert "hakushi_anima_gt_v1" not in node.weight_names()
    w = node.load_auto_weights("hakushi_anima_gt_v1")
    assert sorted(int(k) for k in w["probe_steps"]) == [8, 12, 16, 20, 30]
    for e in w["probe_steps"].values():
        assert len(e["weights"]) == w["blocks"] * len(e["windows"]) + 1
        assert e["feature_step"] < min(int(k) for k, v in w["probe_steps"].items() if v is e)


def test_auto_mask_splits_left_right_and_feature_recording():
    torch.manual_seed(0)
    gh, gw, k = 4, 6, 8
    S = gh * gw
    rec = node.Recorder(["1girl", "1man"], [[0], [1]], keep_blocks=True, keep_features=True,
                        feature_steps={3}, feature_blocks={0, 1})
    rec.grid = (1, gh, gw)
    left = (torch.arange(S) % gw < gw // 2).float()
    share = torch.stack([left, 1 - left]) + 0.01 * torch.rand(2, S)
    for idx in range(k):
        rec.progress[idx] = idx / 30
        for b in range(2):
            rec.add(idx, share, block=b)
            if idx == 3:
                rec.add_feature(idx, b, torch.randn(S, 80) + 3 * left[:, None])
    assert set(rec.features) == {(3, 0), (3, 1)}
    weights = {"blocks": 2, "feature_blocks": [0, 1], "parts": 4, "position_weight": 3.0, "slic_segments": 20,
               "probe_steps": {"8": {"windows": [[0, 8]], "feature_step": 3, "weights": [1.0, 1.0, 0.0],
                                     "loso": {"window": "all"}}}}
    img = torch.zeros(1, 32, 48, 3)
    img[0, :, :24, 0] = 1.0
    img[0, :, 24:, 2] = 1.0
    for backend in ("cpu", "gpu"):   # gpu = torch on the sampling device (cpu torch when there is no GPU)
        label, info = node.auto_label(rec, img[0], weights, backend=backend, segmentation_method="existing")
        assert info["probe_steps"] == 8 and info["feature_step"] == 3
        assert (label[:, :24] == 0).all() and (label[:, 24:] == 1).all(), backend
    pytest.importorskip("maxflow")   # the production default is the min-cut path
    label, info = node.auto_label(rec, img[0], weights)
    assert info["segmentation"] == "graph_cut" and info["graph_nodes"] > 1
    assert (label[:, :24] == 0).all() and (label[:, 24:] == 1).all()
    panel = node.SnapRefRegionAutoMaskPanel().apply(img, img, torch.from_numpy(label == 0).float()[None],
                                                    torch.from_numpy(label == 1).float()[None], caption="x")[0]
    assert panel.shape[2] == 48 * 4 and panel.shape[1] > 32


def test_reference_affinity_drives_the_auto_mask():
    """Person-term attention says the opposite split; source=reference follows the recorded reference shares."""
    torch.manual_seed(0)
    gh, gw, k = 4, 6, 8
    S = gh * gw
    rec = node.Recorder(["1girl", "1man"], [[0], [1]], keep_blocks=True, keep_features=True,
                        feature_steps={3}, feature_blocks={0, 1})
    rec.grid = (1, gh, gw)
    left = (torch.arange(S) % gw < gw // 2).float()
    wrong = torch.stack([1 - left, left])                   # attention: person 1 on the right
    for idx in range(k):
        rec.progress[idx] = idx / 30
        for b in range(2):
            rec.add(idx, wrong, block=b)
            if idx == 3:
                rec.add_feature(idx, b, torch.randn(S, 80) + 3 * left[:, None])
            # log-sum-exp per reference [B, H, S, 1]: reference 1 stronger on the left half
            lse1 = (4 * left - 2).view(1, 1, S, 1).repeat(1, 3, 1, 1)
            rec.add_affinity(idx, [lse1, -lse1])
    aff = rec.affinity_map()
    assert aff.shape == (2, S) and torch.allclose(aff.sum(0), torch.ones(S))
    assert (aff[0][left == 1] > 0.9).all() and (aff[0][left == 0] < 0.1).all()
    weights = {"blocks": 2, "feature_blocks": [0, 1], "parts": 4, "position_weight": 3.0, "slic_segments": 20,
               "probe_steps": {"8": {"windows": [[0, 8]], "feature_step": 3, "weights": [1.0, 1.0, 0.0],
                                     "loso": {"window": "all"}}}}
    img = torch.zeros(1, 32, 48, 3)
    img[0, :, :24, 0] = 1.0
    img[0, :, 24:, 2] = 1.0
    by_attention, _ = node.auto_label(rec, img[0], weights, segmentation_method="existing")
    by_reference, info = node.auto_label(rec, img[0], weights, source="reference", segmentation_method="existing")
    assert (by_attention[:, :24] == 1).all()                 # follows the (wrong) person-term attention
    assert info["source"] == "reference"
    assert (by_reference[:, :24] == 0).all() and (by_reference[:, 24:] == 1).all()
    empty = node.Recorder(["a", "b"], [[0], [1]])
    try:
        node.affinity_score(empty, 1, gh, gw, torch.device("cpu"))
        assert False, "expected an error without recorded affinity"
    except RuntimeError as e:
        assert "ref_affinity" in str(e)


def test_combined_score_flips_only_weak_person_term_regions():
    att = torch.tensor([[3.0, 2.5, -0.2, -3.0, -2.5, -2.8]])        # token 2 weakly person 2 (a hat)
    aff = torch.tensor([[0.60, 0.40, 0.90, 0.20, 0.25, 0.22]])      # token 1 looks a bit like reference 2, token 2 like 1
    score, info = node.combined_score(att, aff, 1.0)
    assert info["ref_weight_used"] == 1.0
    assert (score[0, :3] > 0).all() and (score[0, 3:] < 0).all()   # the hat joins person 1, token 1 stays
    same, info0 = node.combined_score(att, aff, 0.0)
    assert ((same > 0) == (att > 0)).all()
    flat, info1 = node.combined_score(att, torch.full_like(aff, 0.5), 1.0)
    assert info1["ref_weight_used"] == 0.0 and torch.equal(flat, att)   # no separation: keep the person terms


def test_layout_positions_are_detected_in_the_prompt():
    prompt = ("2girls, cafe, the girl on the left has a silver bob, the man standing on the right wears a suit, "
              "the cat at the top of the shelf")
    assert node.term_position("girl on the left") == "left"
    assert node.term_position("1girl | girl", prompt) == "left"
    assert node.term_position("1man | man", prompt) == "right"
    assert node.term_position("1girl | girl", "the left girl smiles") == "left"
    assert node.term_position("1girl | girl", "1girl, 1man, park") is None            # no position stated
    assert node.term_position("1girl | girl", "girl on the left, girl on the right") is None   # ambiguous


def test_instance_embedding_separates_tokens_that_attend_to_each_other():
    torch.manual_seed(0)
    S, H, D = 40, 2, 8
    group = torch.arange(S) % 2 == 0                       # interleaved: position does not tell the groups apart
    centre = torch.randn(2, H, D) * 3
    q = centre[group.long()] + 0.1 * torch.randn(S, H, D)
    rec = node.Recorder(["a", "b"], [[0], [1]])
    rec.add_instance(5, 12, q, q.clone())
    emb = rec.instance[(5, 12)].float()
    assert emb.shape == (S, node.INSTANCE_DIM)
    first = emb[:, 0]
    assert ((first[group] > 0).all() and (first[~group] < 0).all()) or ((first[group] < 0).all() and (first[~group] > 0).all())


def test_instance_vote_keeps_a_person_together():
    torch.manual_seed(0)
    gh, gw = 4, 8
    score = torch.ones(gh, gw)
    score[:, 4:] = -1.0
    score[0, 4] = -1.5                       # a head lying in the other half: its own score says person 2
    person1 = torch.zeros(gh, gw, dtype=torch.bool)
    person1[:, :4] = True
    person1[0, 4] = True                     # ... but it attends with person 1's tokens
    emb = torch.zeros(gh * gw, node.INSTANCE_K)
    emb[person1.flatten(), 0] = 1.0
    emb[~person1.flatten(), 1] = 1.0
    out = node.instance_vote(score, emb + 0.01 * torch.randn_like(emb))
    assert out[0, 4] > 0 and (out[person1] > 0).all() and (out[~person1] < 0).all()


def test_fitted_terms_decide_the_block_weights():
    rec = types.SimpleNamespace(terms=["1girl | girl", "1man | man"])
    assert node.fitted_terms(rec)
    rec.terms = ["lying girl", "sitting girl"]
    assert not node.fitted_terms(rec)
    rec.terms = ["girl on the left", "girl on the right"]
    assert not node.fitted_terms(rec)


def test_graph_cut_separates_two_uniform_regions():
    pytest.importorskip("maxflow")
    torch.manual_seed(0)
    H = W = 64
    img = torch.ones(H, W, 3)
    img[:, : W // 2] = torch.tensor([0.9, 0.1, 0.1])
    img[:, W // 2:] = torch.tensor([0.1, 0.1, 0.9])
    sp = (torch.arange(H)[:, None] // 8 * 8 + torch.arange(W)[None, :] // 8 + 1).long()
    xs = torch.arange(W)[None, :].expand(H, W)
    score = torch.where(xs < W // 2, 1.0, -1.0).contiguous() + 0.05 * torch.randn(H, W)
    label, info = node.graph_cut_refine(img, sp, score)
    assert label.shape == (H, W) and set(label.unique().tolist()) <= {0, 1}
    assert (label[:, : W // 2] == 0).all() and (label[:, W // 2:] == 1).all()
    assert info["graph_nodes"] == 64 and info["graph_edges"] > 0


def test_graph_cut_uniform_scores_and_lambda_zero():
    pytest.importorskip("maxflow")
    torch.manual_seed(2)
    H = W = 64
    img = torch.full((H, W, 3), 0.5)
    sp = (torch.arange(H)[:, None] // 8 * 8 + torch.arange(W)[None, :] // 8 + 1).long()
    label, _ = node.graph_cut_refine(img, sp, torch.zeros(H, W))   # all-equal costs: must not raise
    assert label.shape == (H, W) and set(label.unique().tolist()) <= {0, 1}
    score = torch.randn(H, W)
    plain, _ = node.graph_cut_refine(img, sp, score, lambda_smooth=0.0)
    assert (plain == node.vote_t(score, sp)).all()                 # lambda=0 = plain per-superpixel sign


def test_graph_cut_keeps_a_strong_unary_superpixel():
    pytest.importorskip("maxflow")
    H = W = 64
    img = torch.full((H, W, 3), 0.5)
    sp = (torch.arange(H)[:, None] // 8 * 8 + torch.arange(W)[None, :] // 8 + 1).long()
    score = -torch.ones(H, W)
    score[8:16, 8:16] = 30.0                                      # one very strong person-1 superpixel
    label, _ = node.graph_cut_refine(img, sp, score, lambda_smooth=0.5)
    keep = torch.zeros(H, W, dtype=torch.bool)
    keep[8:16, 8:16] = True
    assert (label[keep] == 0).all() and (label[~keep] == 1).all()


def test_graph_cut_ignores_superpixel_numbering():
    pytest.importorskip("maxflow")
    torch.manual_seed(1)
    H = W = 64
    img = torch.rand(H, W, 3) * 0.3 + 0.35
    sp = (torch.arange(H)[:, None] // 8 * 8 + torch.arange(W)[None, :] // 8 + 1).long()
    score = torch.randn(H, W) * torch.linspace(-1, 1, W)[None, :]
    base, _ = node.graph_cut_refine(img, sp, score)
    perm = torch.randperm(int(sp.max())) + 1
    shuffled = perm[sp - 1]
    again, _ = node.graph_cut_refine(img, shuffled, score)
    assert (base == again).all()


def test_graph_cut_nonlocal_follows_self_attention():
    pytest.importorskip("maxflow")
    H = W = 64
    img = torch.full((H, W, 3), 0.5)
    sp = (torch.arange(H)[:, None] // 8 * 8 + torch.arange(W)[None, :] // 8 + 1).long()
    xs = torch.arange(W)[None, :].expand(H, W)
    score = torch.where(xs < W // 2, 1.0, -1.0).contiguous()
    score[:, 40:48] = 1.0                                        # one column that disagrees with the true split
    plain, _ = node.graph_cut_refine(img, sp, score, lambda_smooth=0.2)
    assert (plain[:, 40:48] == 0).all()                          # without non-local edges it stays wrong
    gh = gw = 16
    tx = (torch.arange(gw)[None, :] * (W // gw)).expand(gh, gw)
    emb = torch.zeros(gh * gw, 2)
    emb[(tx < W // 2).reshape(-1), 0] = 1.0                     # self-attention clusters = the true halves
    emb[(tx >= W // 2).reshape(-1), 1] = 1.0
    fixed, info = node.graph_cut_refine(img, sp, score, lambda_smooth=0.2, emb=emb,
                                        token_grid=(gh, gw), nonlocal_k=4, nonlocal_weight=10.0)
    assert info["nonlocal_edges"] > 0
    assert (fixed[:, : W // 2] == 0).all() and (fixed[:, W // 2:] == 1).all()



def test_instance_vote_splits_bimodal_conflict_cluster():
    torch.manual_seed(0)
    gh, gw = 8, 8
    S = gh * gw
    # First 4 dims merge person 1 (rows 0..3) and person 2 (rows 4..7) into the same cluster; dim 5 separates them
    emb8 = torch.zeros(S, 8)
    emb8[:, 0] = 1.0
    emb8[:32, 5] = 1.0
    emb8[32:, 5] = -1.0
    emb8 += 0.01 * torch.randn_like(emb8)
    emb4 = torch.nn.functional.normalize(emb8[:, :4], dim=1)
    score = torch.ones(gh, gw)
    score[4:, :] = -1.5
    fg = torch.ones(gh, gw)
    out = node.instance_vote(score, emb4, fg_weight=fg, emb_full=emb8)
    assert (out[:4, :] > 0).all() and (out[4:, :] < 0).all()


def test_auto_mask_half_resolution_and_skip_features_when_instance():
    pytest.importorskip("maxflow")
    torch.manual_seed(0)
    spec_opt = node.SnapRefRegionAutoMask.INPUT_TYPES()["optional"]
    assert "half_resolution" in spec_opt and spec_opt["half_resolution"][1]["default"] is True
    gh, gw, k = 8, 8, 8
    S = gh * gw
    rec = node.Recorder(["1girl", "1man"], [[0], [1]], keep_blocks=True, keep_features=False)
    rec.grid = (1, gh, gw)
    left = (torch.arange(S) % gw < gw // 2).float()
    share = torch.stack([left, 1 - left])
    for idx in range(k):
        rec.progress[idx] = idx / 30
        for b in range(2):
            rec.add(idx, share, block=b)
    emb = torch.zeros(S, node.INSTANCE_DIM)
    emb[left == 1, 0] = 1.0
    emb[left == 0, 1] = 1.0
    rec.instance[(8, node.INSTANCE_AUTO_BLOCK)] = emb.half()
    weights = {"blocks": 2, "feature_blocks": [0, 1], "parts": 4, "position_weight": 3.0, "slic_segments": 40,
               "probe_steps": {"8": {"windows": [[0, 8]], "feature_step": 3, "weights": [1.0, 1.0, 0.0],
                                     "loso": {"window": "all"}}}}
    img = torch.zeros(128, 128, 3)
    img[:, :64, 0] = 1.0
    img[:, 64:, 2] = 1.0
    lab_half, info_half = node.auto_label(rec, img, weights, half_resolution=True)
    lab_full, info_full = node.auto_label(rec, img, weights, half_resolution=False)
    assert info_half["half_resolution"] is True and info_full["half_resolution"] is False
    assert (lab_half[:, :60] == 0).all() and (lab_half[:, 68:] == 1).all()
    assert (lab_full[:, :60] == 0).all() and (lab_full[:, 68:] == 1).all()


def test_fast_add_instance_large_sequence():
    torch.manual_seed(0)
    S, H, D = 300, 4, 8
    group = torch.arange(S) % 2 == 0
    centre = torch.randn(2, H, D) * 3
    q = centre[group.long()] + 0.1 * torch.randn(S, H, D)
    rec = node.Recorder(["a", "b"], [[0], [1]])
    rec.add_instance(5, 16, q, q.clone())
    emb = rec.instance[(5, 16)].float()
    assert emb.shape == (S, node.INSTANCE_DIM)
    first = emb[:, 0]
    assert ((first[group] > 0).all() and (first[~group] < 0).all()) or ((first[group] < 0).all() and (first[~group] > 0).all())


def test_instance_vote_recovers_bridged_displaced_part():
    torch.manual_seed(0)
    gh, gw = 10, 12
    S = gh * gw
    emb8 = torch.zeros(S, 8)
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    p1_upper = ((ys < 5) & (xs < 5)).flatten()
    p1_lower = ((ys >= 5) & (xs < 5)).flatten()
    p1_head_left = ((ys >= 3) & (ys <= 6) & (xs == 5)).flatten()
    p1_head_right = ((ys >= 3) & (ys <= 6) & (xs >= 6) & (xs <= 7)).flatten()
    bg = ((ys < 2) & (xs >= 5)).flatten()
    p2_body = ~(p1_upper | p1_lower | p1_head_left | p1_head_right | bg)
    # 4 clusters in first 4 dims: c0 = p1_upper + p1_lower + p1_head_left (54 tokens: 50 pos, 4 neg -> neg_sh = 4/54 = 0.074.. wait: make 7 tokens in p1_head_left so 7/57 = 0.123!)
    p1_head_left = ((ys >= 2) & (ys <= 8) & (xs == 5)).flatten()
    p1_head_right = ((ys >= 2) & (ys <= 8) & (xs == 6)).flatten()
    bg = ((ys < 2) & (xs >= 5)).flatten()
    p2_a = ((ys >= 2) & (ys < 6) & (xs >= 7)).flatten()
    p2_b = ((ys >= 6) & (xs >= 7)).flatten()
    emb8[p1_upper, 0] = 1.0
    emb8[p1_upper, 4] = 0.8
    emb8[p1_lower, 0] = 1.0
    emb8[p1_lower, 4] = -0.8
    emb8[p1_head_left, 0] = 1.0
    emb8[p1_head_left, 5] = 1.0
    emb8[p1_head_right, 2] = 0.9
    emb8[p1_head_right, 5] = 1.2
    emb8[bg, 1] = 1.0
    emb8[p2_a, 2] = 1.0
    emb8[p2_b, 3] = 1.0
    emb8 += 0.01 * torch.randn_like(emb8)
    emb4 = torch.nn.functional.normalize(emb8[:, :4], dim=1)
    score = torch.where(xs < 5, 1.0, -1.0).float()
    fg = torch.ones(gh, gw)
    out = node.instance_vote(score, emb4, fg_weight=fg, emb_full=emb8)
    assert (out[2:9, 5:7] > 0).all()
