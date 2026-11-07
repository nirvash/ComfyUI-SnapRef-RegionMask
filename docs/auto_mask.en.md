# How the automatic masks work

[日本語](auto_mask.ja.md) · [README](../README.md#automatic-masks-experimental)

How SnapRef Region Probe and SnapRef Region Auto Mask estimate each person's region, and how the weights are fitted. See the README for usage.

## Pipeline

1. **Record** (Region Probe): at every step, from the cross-attention of every model block (28 in Anima), how much each image token attends to the `person_1` / `person_2` terms of the prompt. At chosen steps and blocks it also keeps the self-attention output (token features), reduced to 64 dimensions by PCA. The sampling result is not changed.
2. **Score** (Auto Mask): per block, the normalized person 1 map minus the person 2 map; the learned per-block weights add them into one score per token (> 0 means person 1). The weights were fitted to `1girl` / `1man`; for any other person terms the plain mean of the blocks is used.
3. **Parts** (when no self-attention instances were recorded): k-means on the token features of one step (blocks 8, 12, 16) plus position, 192 parts. Each part takes the sign of its summed score, which turns the token-level speckle into coherent parts.
4. **Pixels**: SLIC splits the estimation image into about 800 superpixels; each receives one label, so the outline follows image edges. With `source = attention`, the torch path mixes the part vote with the original token score so local evidence can survive the part vote.
5. Output: `mask_1` (person 1) and `mask_2` (person 2). Every pixel goes to one of them, background included.

The GPU path uses grid-initialized k-means with a fixed number of rounds and a torch implementation of SLIC. The CPU path (scikit-image) for `source = attention` keeps the original majority vote.


## Instances from self-attention

Tokens of one person attend to each other while the image is generated, which is how hair, face and clothes stay consistent. With `instance_mask` the probe takes block 16's self-attention at one step (step 8, or the last step of a shorter pass), averages the heads into a token-to-token matrix (about 4000 × 4000 at 1 MP), and keeps the first eigenvectors of its normalized form (a spectral embedding, 8 dimensions). Auto Mask clusters the first 4 dimensions with k-means (4 clusters, 8 random starts, the lowest cost kept). Each token then takes its cluster's mean person score plus 0.3 of its own score, and SLIC snaps the outline to image edges. Recording costs about 0.08 s per pass, estimating about 0.1 s.

This keeps a person together where the person terms alone fail: the lying girl's head on the other girl's lap attends to the person term of the side it lies on, but to its own body in self-attention. On six two-girl scenes and a lap pillow (no references, 30 steps, hand-traced masks) the person error went from 2.2% / 8.8% to 1.3% / 4.5%.

## Stages

- **Two stages (default)**: an 8-step pass without references gives the masks; the final pass reads each reference in its region. By step 8 the composition is settled and the instances are clear; at 4 steps they are not (37% on one lap-pillow image). This needs a prompt that fixes who is where (see the README).
- **Three stages (`refine`)**: a 4-step draft, then SnapRef with the draft masks for steps 0–8 and the masks estimated again from that pass's own recording and its x0 prediction (`denoised_output`). Useful when the references move the composition. The refine pass samples part of the 30-step schedule (SplitSigmas), so SnapRef Region Mask's `ref_start` is rescaled to the segment (0.1 over 30 steps → 0.3125 over 8 steps).

## Per-block weights (`weights/*.json`)

For each draft length k (8, 12, 16, 20, 30):

| Field | Content |
|---|---|
| `windows` | step windows of the score (all of 0..k, its last third, or both) |
| `weights` | one weight per window and block, plus the intercept (ridge regression) |
| `feature_step` | the step whose token features make the parts |
| `loso` | error when each scene is left out of the fit (below) |

When the recorded step count does not match a k, the largest k below it is used.

### Bundled `hakushi_anima_gt_v1`

- **Data**: 29 images, hakushiMixAnima v1.0, 30 steps (er_sde / simple, CFG 4), no SnapRef, person terms `1girl | girl` and `1man | man`.
- **Ground truth**: per pixel 0 = person 1, 1 = person 2, 255 = other; the estimates were corrected by hand and checked by eye.
- **Metric**: the share of person pixels given to the other person (background leaks are not counted), fitted with each scene (same prompt) left out and measured on that scene.
- **Result** (mean): 7.6% at k=8, 4.1% at k=12, 3.4% at k=20, 3.6% at k=30.
- The data has no SnapRef, while the refine stage that estimates in practice has it; the same weights work there.

### Fitting your own

Use `tools/fit_auto_mask_weights.py`.

1. **Generate and record** two-person scenes without SnapRef for 30 steps, with Region Probe's `auto_mask` and `dump_path` set (per-block maps and token features go to a `.npz`). Two or three seeds per prompt allow the per-scene evaluation.
2. **Ground truth**: an 8-bit PNG of the image size (0 = person 1, 1 = person 2, 255 = other). Correcting Auto Mask's own output is the quickest way.
3. **Index**: `index.json` in the dataset folder:

```json
[
  {"id": "bench-01", "scene": "bench", "image": "images/bench-01.png", "dump": "dumps/bench-01.npz", "gt": "gt/bench-01.png"}
]
```

4. **Fit**:

```sh
python tools/fit_auto_mask_weights.py <dataset> --out weights/my_weights.json --description "..."
```

Each window mode and feature step is scored leaving one scene out; the best one is refitted on all images and written to `--out`, with the scores in `my_weights.report.json`. Files in `weights/` appear in Region Probe's `auto_mask` and Auto Mask's `weights` after a ComfyUI restart.

## Resemblance to the references (`source`)

With the person terms alone, parts the prompt gives to only one person (a beret, a bunny-ears hood) often go to the other person or the background: the person terms attend to the body, weakly to such parts.

- SnapRef Region Mask computes self-attention per reference. With Region Probe's `ref_affinity` on, it hands over the per-reference attention strength (log-sum-exp) before the region bias, and the probe keeps the share of reference 1 per token. SnapRef computes this anyway, so it costs about nothing.
- `source = reference` splits by that share alone. It fails on its own: a whole picture tends to lean to one reference, and features both references share (a white blouse, the face) carry no signal.
- `source = both` lets the person terms decide the bodies and the resemblance correct them:
  1. The person-term score gives provisional person 1 / person 2 regions.
  2. The mean share of reference 1 over each region is computed; their midpoint becomes 0 and the means ±1 (clamped at ±3), which removes the per-image lean.
  3. The person-term score, divided by its spread, plus `ref_weight` times the centred resemblance.
  - Confident person-term regions (faces, bodies) stay; regions the person terms miss but the resemblance clearly assigns (hats, hoods) flip.
  - `ref_weight` 0.5–1 was balanced; from 2 on, parts of the other person start to join.

## Limits

- The fitted person terms are `1girl` / `1man`. Two girls worked with position words (`girl on the left` / `girl on the right`); hair or clothing words as person terms pick up only those items and leave most of the body to the other person.
- Adding item words to a person term with `|` hurts: their attention is concentrated on the item and crowds out the body. Use `source = both` for items.
- Two people only. Three would need per-person scores, a three-way vote and fitting on three-person scenes.
- The masks only limit where each reference is read; prompt words and self-attention between image tokens still reach both people.
