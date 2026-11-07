"""Unit tests that run without ComfyUI: python -m pytest tests"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

# minimal stand-in for comfy.patcher_extension so the node module imports outside ComfyUI
comfy = types.ModuleType("comfy")
pe = types.ModuleType("comfy.patcher_extension")
pe.WrappersMP = types.SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
comfy.patcher_extension = pe
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.patcher_extension", pe)

spec = importlib.util.spec_from_file_location("snapref_region_mask", Path(__file__).resolve().parents[1] / "__init__.py")
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


def test_box_mask_hard_edges():
    m = node.box_mask(4, 8, (0.0, 0.0, 0.5, 1.0), feather=0.0, outside=0.0)
    assert m.shape == (4, 8)
    assert torch.all(m[:, :4] == 0)
    assert torch.all(torch.isneginf(m[:, 4:]))


def test_box_mask_outside_strength():
    m = node.box_mask(2, 4, (0.5, 0.0, 0.5, 1.0), feather=0.0, outside=0.25)
    assert torch.allclose(m[:, :2], torch.full((2, 2), math.log(0.25)))
    assert torch.all(m[:, 2:] == 0)


def test_box_mask_feather_is_monotonic_across_edge():
    m = node.box_mask(1, 40, (0.0, 0.0, 0.5, 1.0), feather=0.2, outside=0.0)[0]
    finite = m[torch.isfinite(m)]
    assert torch.all(finite[1:] <= finite[:-1] + 1e-6)   # 0 inside, falling towards the edge


def test_box_mask_no_feather_on_image_border():
    m = node.box_mask(1, 40, (0.0, 0.0, 0.5, 1.0), feather=0.2, outside=0.0)[0]
    assert m[0] == 0   # the leftmost token lies on the image border, not on the box edge


def test_mask_grid_resizes_and_maps():
    mask = torch.zeros(1, 64, 128)
    mask[:, 32:, :] = 1.0                      # bottom half
    g = node.mask_grid(mask, 4, 8, outside=0.0)
    assert g.shape == (4, 8)
    assert torch.all(torch.isneginf(g[:2])) and torch.all(g[2:] == 0)
    g = node.mask_grid(mask[0], 4, 8, outside=0.5)   # [H, W] also accepted
    assert torch.allclose(g[:2], torch.full((2, 8), math.log(0.5)))


def test_mask_overrides_box():
    m = node.SnapRefRegionMask()
    spec = node.SnapRefRegionMask.INPUT_TYPES()["optional"]
    assert spec["mask_1"][0] == "MASK" and spec["mask_2"][0] == "MASK"
    mask = torch.ones(1, 8, 8)
    regions = []

    def fake_wrapper(r, *a):
        regions.extend(r)
        return lambda *args, **kw: None

    orig, node._make_wrapper = node._make_wrapper, fake_wrapper
    try:
        model = types.SimpleNamespace(clone=lambda: types.SimpleNamespace(
            wrappers={}, add_wrapper_with_key=lambda t, k, w: None))
        m.apply(model, layout="left_right", split=0.5, mask_1=mask, mask_2=None)
    finally:
        node._make_wrapper = orig
    assert regions[0][0] == "mask" and regions[1] == ("box", (0.5, 0.0, 0.5, 1.0))


def test_layout_boxes():
    b1, b2 = (0.1, 0.2, 0.3, 0.4), (0.6, 0.0, 0.0, 1.0)
    assert node.layout_boxes("left_right", 0.4, b1, b2) == [(0.0, 0.0, 0.4, 1.0), (0.4, 0.0, 0.6, 1.0)]
    assert node.layout_boxes("top_bottom", 0.25, b1, b2) == [(0.0, 0.0, 1.0, 0.25), (0.0, 0.25, 1.0, 0.75)]
    assert node.layout_boxes("boxes", 0.5, b1, b2) == [b1]                 # width_2 = 0: box 1 only
    assert node.layout_boxes("boxes", 0.5, b1, (0.5, 0, 0.5, 1)) == [b1, (0.5, 0, 0.5, 1)]


def test_sampling_progress():
    sched = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert node.sampling_progress({"sigmas": torch.tensor([1.0]), "sample_sigmas": sched}) == 0.0
    assert node.sampling_progress({"sigmas": torch.tensor([0.5]), "sample_sigmas": sched}) == 0.5
    assert node.sampling_progress({}) is None


def _dm(with_ref):
    attn = types.SimpleNamespace(_ref_kv_state={"cache": []} if with_ref else None)
    return types.SimpleNamespace(blocks=[types.SimpleNamespace(self_attn=attn)])


def test_split_markers_rewritten_for_snapref_halves():
    x = torch.zeros(1, 16, 1, 8, 8)
    base = {"cond_or_uncond": [1, 0], node.FULL_BATCH_KEY: 2}
    kw = {"transformer_options": dict(base)}
    node._fix_split_markers(_dm(True), x, kw)
    assert kw["transformer_options"]["cond_or_uncond"] == [0]
    kw = {"transformer_options": dict(base)}
    node._fix_split_markers(_dm(False), x, kw)
    assert kw["transformer_options"]["cond_or_uncond"] == [1]


def test_unsplit_batch_keeps_markers():
    x = torch.zeros(2, 16, 1, 8, 8)
    kw = {"transformer_options": {"cond_or_uncond": [1, 0], node.FULL_BATCH_KEY: 2}}
    node._fix_split_markers(_dm(True), x, kw)
    assert kw["transformer_options"]["cond_or_uncond"] == [1, 0]


def test_attach_restores_snapref_forward():
    def snapref_forward(x, context=None, rope_emb=None, transformer_options={}):
        return "snapref"

    attn = types.SimpleNamespace(_ref_kv_state=None, forward=snapref_forward)
    plain = types.SimpleNamespace(forward="untouched")   # block without SnapRef's hook
    dm = types.SimpleNamespace(blocks=[types.SimpleNamespace(self_attn=attn), types.SimpleNamespace(self_attn=plain)])
    restore = node._attach(dm)
    assert attn.forward is not snapref_forward
    assert attn.forward(torch.zeros(1)) == "snapref"   # no mask active: passes straight through
    assert plain.forward == "untouched"
    node._detach(restore)
    assert attn.forward is snapref_forward


def test_node_inputs():
    spec = node.SnapRefRegionMask.INPUT_TYPES()
    assert {"model", "layout", "split", "feather", "outside_strength", "ref_start"} <= set(spec["required"])
    assert spec["required"]["layout"][0] == node.LAYOUTS
    assert {"x", "y", "width", "height", "x_2", "y_2", "width_2", "height_2", "mask_1", "mask_2"} <= set(spec["optional"])


def test_masked_attention_reports_reference_affinity():
    """With SnapRef Region Probe's callback in transformer_options, the hooked self-attention reports one
    log-sum-exp per reference (before the region bias). Needs CUDA (the masked path runs in fp16 there)."""
    if not torch.cuda.is_available():
        return
    dev, dt = torch.device("cuda"), torch.float16
    B, S, H, D, N = 1, 16, 2, 8, 4
    torch.manual_seed(0)
    k_r = torch.randn(1, 2 * N, H, D, device=dev, dtype=dt)
    v_r = torch.randn(1, 2 * N, H, D, device=dev, dtype=dt)

    class Attn:
        def __init__(self):
            self._ref_kv_state = {"cache": [(k_r, v_r)], "sizes": [N, N], "biases": [0.0, 0.0]}
            self.forward = lambda *a, **k: None

        def compute_qkv(self, x, context, rope_emb=None, transformer_options={}):
            q = x.view(B, S, H, D)
            return q, q.clone(), q.clone()

        def output_proj(self, t):
            return t

        def output_dropout(self, t):
            return t

    attn = Attn()
    dm = types.SimpleNamespace(blocks=[types.SimpleNamespace(self_attn=attn)])
    node._REGION["bias"] = [torch.zeros(S, device=dev), torch.full((S,), float("-inf"), device=dev)]
    got = []
    restore = node._attach(dm)
    try:
        out = attn.forward(torch.randn(B, S, H * D, device=dev, dtype=dt), None,
                           transformer_options={node.AFFINITY_KEY: lambda block, lses: got.append((block, lses))})
    finally:
        node._detach(restore)
        node._REGION["bias"] = None
    assert out.shape == (B, S, H * D)
    assert len(got) == 1 and got[0][0] == 0 and len(got[0][1]) == 2
    assert all(l.shape == (B, H, S, 1) and torch.isfinite(l).all() for l in got[0][1])   # bias not applied
