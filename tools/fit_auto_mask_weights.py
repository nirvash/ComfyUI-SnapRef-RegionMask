"""Fit the per-block weights of SnapRef Region Auto Mask (weights/*.json with probe_steps) to hand-checked masks.

Input: a dataset folder with index.json, a list of
  {"id": "bench-01", "scene": "bench", "image": "images/bench-01.png", "dump": "dumps/bench-01.npz",
   "gt": "gt/bench-01.png"}
paths relative to the folder.
  image  the generated image (the one the ground truth was drawn on)
  dump   SnapRef Region Probe's dump_path .npz of that generation (per-block maps and token features; record the
         whole schedule, e.g. 30 steps, so every probe pass length k below can be cut from it)
  gt     8-bit PNG, same size as the image: 0 = person 1, 1 = person 2, 255 = anything else
  scene  images of one scene (same prompt) share it; the score is measured leaving one scene out

For each probe pass length k (--ks): the score is a ridge regression over per-block features (normalized person 1 map
minus person 2 map, mean over a step window inside 0..k: all, last third, or both), the parts come from k-means on the
token features of one recorded step < k, and the label is voted per part and then per SLIC superpixel of the image,
exactly as the node does it. The window mode and feature step with the lowest leave-one-scene-out error are kept and
the weights are refitted on all images.

  python tools/fit_auto_mask_weights.py <dataset> --out weights/my_weights.json [--description "..."]
Needs numpy, torch, Pillow, scikit-image (the ComfyUI environment has them). Writes a report next to --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic

FEATURE_BLOCKS = (8, 12, 16)
PARTS = 192
POSITION_WEIGHT = 3.0
SLIC_SEGMENTS = 800


def normalize(maps, low=0.02, high=0.995):
    flat = maps.reshape(maps.shape[0], -1).astype(np.float64)
    lo = np.quantile(flat, low, axis=1)[:, None]
    hi = np.quantile(flat, high, axis=1)[:, None]
    return np.clip((flat - lo) / np.maximum(hi - lo, 1e-12), 0, 1)


def block_features(per_block, a, b):
    """[S, B]: per block, normalized person 1 map minus person 2 map, mean over steps a..b-1."""
    maps = per_block[a:b].astype(np.float32).mean(axis=0)
    return np.stack([normalize(maps[i])[0] - normalize(maps[i])[1] for i in range(maps.shape[0])], axis=1)


def windows(k, mode):
    tail = (k - max(1, k // 3), k)
    return {'all': [(0, k)], 'tail': [tail], 'both': [(0, k), tail]}[mode]


def ridge(X, y, lam=1.0):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ y)


def grid_gt(gt, gh, gw):
    """Majority label per token cell: 0 person 1, 1 person 2, 255 other (person pixels < 50 % of the cell)."""
    h, w = gt.shape
    out = np.full((gh, gw), 255, dtype=np.uint8)
    for y in range(gh):
        for x in range(gw):
            cell = gt[y * h // gh:(y + 1) * h // gh, x * w // gw:(x + 1) * w // gw]
            p1, p2 = (cell == 0).mean(), (cell == 1).mean()
            if p1 + p2 >= 0.5:
                out[y, x] = 0 if p1 >= p2 else 1
    return out


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(X).float()
    c = [X[torch.randint(len(X), (1,), generator=g)].squeeze(0)]
    for _ in range(k - 1):
        d = torch.cdist(X, torch.stack(c)).min(1).values ** 2
        c.append(X[torch.multinomial(d / d.sum(), 1, generator=g)].squeeze(0))
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
    return torch.cdist(X, c).argmin(1).numpy()


def token_parts(feats, fsteps, fblocks, step, gh, gw):
    chunks = []
    for b in FEATURE_BLOCKS:
        x = feats[list(fsteps).index(step), list(fblocks).index(b)].astype(np.float32)
        chunks.append(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6) / np.sqrt(len(FEATURE_BLOCKS)))
    ys, xs = np.mgrid[0:gh, 0:gw]
    pos = np.stack([ys.ravel() / max(gh, gw), xs.ravel() / max(gh, gw)], axis=1) * POSITION_WEIGHT
    return kmeans(np.hstack(chunks + [pos]), PARTS).reshape(gh, gw) + 1


def upsample(score, h, w):
    t = torch.from_numpy(score.astype(np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0, 0].numpy()


def vote(score, parts):
    lab = (score <= 0).astype(np.uint8)
    ids = parts.ravel()
    sums = np.bincount(ids, weights=score.ravel(), minlength=int(ids.max()) + 1)
    out = (sums <= 0).astype(np.uint8)[ids].reshape(parts.shape)
    out[parts == 0] = lab[parts == 0]
    return out


def cross_error(lab, gt):
    person = gt != 255
    return float((lab[person] != gt[person]).mean())


def load(root):
    runs = []
    for e in json.loads((root / 'index.json').read_text(encoding='utf-8')):
        with np.load(root / e['dump']) as d:
            dump = {k: d[k] for k in d.files}
        if 'features' not in dump:
            raise SystemExit(f"{e['dump']}: no token features; record it with dump_path set")
        _, gh, gw = (int(v) for v in dump['grid'])
        gt = np.array(Image.open(root / e['gt']))
        img = np.array(Image.open(root / e['image']).convert('RGB'))
        if gt.shape != img.shape[:2]:
            raise SystemExit(f"{e['id']}: ground truth and image sizes differ")
        runs.append({'id': e['id'], 'scene': e.get('scene', e['id']), 'gt': gt, 'grid': (gh, gw),
                     'pb': dump['per_block'], 'gtgrid': grid_gt(gt, gh, gw).ravel(), 'f': dump['features'],
                     'fs': [int(s) for s in dump['feature_steps']], 'fb': [int(b) for b in dump['feature_blocks']],
                     'sp': slic(img, n_segments=SLIC_SEGMENTS, compactness=10, start_label=1).astype(np.int64)})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ks', default='8,12,16,20,30', help='probe pass lengths to fit (steps)')
    ap.add_argument('--description', default='')
    a = ap.parse_args()
    runs = load(a.dataset)
    ks = [int(k) for k in a.ks.split(',')]
    fsteps = runs[0]['fs']
    shipped, report = {}, {}
    print(f'{"k":>3s} {"window":6s} {"feat":>4s}  mean   median  OK<=10%  max')
    for k in ks:
        best = None
        for mode in ('all', 'tail', 'both'):
            wins = windows(k, mode)
            for r in runs:
                r['X'] = np.hstack([block_features(r['pb'], s, e) for s, e in wins])
            scores = {}
            for sc in sorted({r['scene'] for r in runs}):
                train = [r for r in runs if r['scene'] != sc]
                X = np.concatenate([r['X'][r['gtgrid'] != 255] for r in train])
                y = np.concatenate([np.where(r['gtgrid'][r['gtgrid'] != 255] == 0, 1.0, -1.0) for r in train])
                w = ridge(X, y)
                for r in runs:
                    if r['scene'] == sc:
                        scores[r['id']] = np.hstack([r['X'], np.ones((len(r['X']), 1))]) @ w
            for fs_ in [s for s in fsteps if s < k][-2:]:
                errs = []
                for r in runs:
                    h, w_ = r['gt'].shape
                    s = upsample(scores[r['id']].reshape(r['grid']), h, w_)
                    parts = token_parts(r['f'], r['fs'], r['fb'], fs_, *r['grid'])
                    parts = np.array(Image.fromarray(parts.astype(np.int32)).resize((w_, h), Image.NEAREST)).astype(np.int64)
                    coarse = vote(s, parts)
                    errs.append(cross_error(vote(np.where(coarse == 0, 1.0, -1.0), r['sp']), r['gt']))
                e = np.array(errs)
                row = {'window': mode, 'feature_step': int(fs_), 'mean': float(e.mean()), 'median': float(np.median(e)),
                       'ok10': float((e <= 0.10).mean()), 'max': float(e.max())}
                report[f'k{k} {mode} f{fs_}'] = row
                print(f'{k:>3d} {mode:6s} {fs_:>4d}  {e.mean():.3f}  {np.median(e):.3f}   {(e <= .1).mean():.2f}    {e.max():.3f}',
                      flush=True)
                if best is None or e.mean() < best[0]:
                    best = (e.mean(), mode, int(fs_))
        if best is None:
            raise SystemExit(f'k={k}: no token features recorded before step {k}')
        _, mode, fs_ = best
        wins = windows(k, mode)
        X = np.concatenate([np.hstack([block_features(r['pb'], s, e) for s, e in wins])[r['gtgrid'] != 255] for r in runs])
        y = np.concatenate([np.where(r['gtgrid'][r['gtgrid'] != 255] == 0, 1.0, -1.0) for r in runs])
        shipped[str(k)] = {'windows': [list(wv) for wv in wins], 'feature_step': fs_,
                           'weights': [round(float(v), 6) for v in ridge(X, y)], 'loso': report[f'k{k} {mode} f{fs_}']}
    a.out.write_text(json.dumps({
        'description': a.description or f'Fitted to {len(runs)} hand-checked masks with tools/fit_auto_mask_weights.py',
        'blocks': int(runs[0]['pb'].shape[1]), 'feature_blocks': list(FEATURE_BLOCKS), 'parts': PARTS,
        'position_weight': POSITION_WEIGHT, 'slic_segments': SLIC_SEGMENTS, 'probe_steps': shipped}, indent=1) + '\n',
        encoding='utf-8')
    a.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
