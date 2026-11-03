# ComfyUI-SnapRef-RegionMask

English | [日本語](README.ja.md)

Assign a region to each [SnapRef](https://github.com/ArcAnima/SnapRef) reference image in Anima to reduce mixing of hair, clothing and accessories between two characters. Supports left/right splits, top/bottom splits, boxes and arbitrary MASKs.

![Top: SnapRef only. Bottom: Region Mask. Same seeds and prompt](docs/comparison.png)

![Picnic, shopping and claw machine](docs/situations.png)

![Chibi characters in left/right and top/bottom layouts](docs/chibi.png)

## Install and use

Requires ComfyUI with Anima support, the SnapRef node and adapter, an Anima model with matching text encoder and VAE, and a CUDA GPU supporting fp16/bf16.

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/nirvash/ComfyUI-SnapRef-RegionMask
```

Restart ComfyUI. Connect **SnapRef → SnapRef Region Mask** between SnapRef's MODEL output and KSampler. Set SnapRef to `multi_layout = separate` with one reference image per character. No regional prompting node is required.

## Samples

Copy images from `examples/input` to `ComfyUI/input`, open a workflow JSON in ComfyUI, and select your model, CLIP, VAE and adapter files.

| Workflow (in examples) | Scene | Output |
|---|---|---|
| `snapref_region_mask_sample_boxes.json` | A on the left, B on the right | `sample_boxes_output.png` |
| `snapref_region_mask_sample_masks.json` | C above, D below, using MASKs | `sample_masks_output.png` |
| `snapref_region_mask_sample_arcade.json` | Swapped references: B left with a triangle mouth, A operating on the right; boxes exclude the machine from reference regions | `sample_arcade_output.png` |

`*_api.json` files are API graphs. Output PNGs also contain their editing workflows. Bypass Region Mask (Ctrl+B) to compare against SnapRef alone with the same settings.

| A | B | C | D |
|---|---|---|---|
| <img src="examples/input/snapref_region_mask_ref_A.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_B.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_C.png" width="160"> | <img src="examples/input/snapref_region_mask_ref_D.png" width="160"> |

## Inputs

Ranges below are the node's UI ranges. Coordinates and dimensions are fractions of the full image.

| Input | Range / choices | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | SnapRef's MODEL output |
| `layout` | `left_right` / `top_bottom` / `boxes` | `left_right` | Split horizontally, vertically, or use boxes. Reference 1 goes left/top; reference 2 right/bottom |
| `split` | 0.05–0.95 | 0.5 | Boundary of the left/right or top/bottom split |
| `feather` | 0–0.5 | 0.05 | Soft edge width for boxes. MASKs retain their own soft edges |
| `outside_strength` | 0–1 | 0 | Reference strength outside its region: 0 disables it, 1 keeps full strength |
| `ref_start` | 0–0.95 | 0.1 | Initial fraction of sampling steps without references |
| `x`, `y` | 0–1 each | 0, 0 | Top-left of reference 1's box |
| `width`, `height` | 0–1 each | 0.5, 1 | Width and height of reference 1's box |
| `x_2`, `y_2` | 0–1 each | 0.5, 0 | Top-left of reference 2's box |
| `width_2`, `height_2` | 0–1 each | 0, 1 | Reference 2's width and height. `width_2 = 0` uses region 1 for all references |
| `mask_1`, `mask_2` | MASK, values 0–1 | disconnected | White reads the reference. Overrides that reference's layout region |

MASKs are resized to the model's token grid. A third reference (`image_3`) uses region 2.

When characters occupy only part of the canvas, use `boxes` to limit references to their regions and leave objects such as the machine outside them.

## Notes

- Briefly describe each character's position, hair, eyes and clothing in the prompt. Examples were tested with SnapRef `strength = 0` (the default).
- Masking requires `fidelity = 1`, `ref_guidance_on = off`, `ref_in_uncond = off`, and CUDA fp16/bf16. Other settings run SnapRef without masking.
- Exact reference fidelity is not guaranteed. Characters crossing region boundaries or overlapping heavily can mix or break. Objects inside a region can also pick up reference features.
- Tested: SnapRef v2a pre2, hakushiMixAnima v1.0, 30 steps, er_sde / simple, CFG 4, 1216×832 / 832×1216.

[Implementation, experiment notes and Anima Regional integration](docs/notes.en.md)

## Tests and license

`python -m pytest tests` (runs without ComfyUI). Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE). SnapRef and its adapter weights have separate licenses.
