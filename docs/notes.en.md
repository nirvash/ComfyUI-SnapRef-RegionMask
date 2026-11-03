# Implementation and integration notes

[README](../README.md)

## How it works

SnapRef prepends the key/value tokens of the reference image to every self-attention call of Anima.
This node adds a per-token bias to the attention scores of each reference:

- inside the reference's region: `0` (read exactly as SnapRef reads it)
- outside: `log(outside_strength)`, i.e. `-inf` by default (not read at all)
- a short ramp (`feather`) softens box edges; a MASK keeps its own soft edges

Inside a region the computation is identical to plain SnapRef. The node does not touch the text prompt
(cross-attention), SnapRef's weights or ComfyUI. Its attention hook is attached for each model call and
removed right after, so the model is left as SnapRef set it up.

For the first `ref_start` fraction of the steps no token reads any reference, so the prompt places the
people first.

## Tips

- **Keep SnapRef `strength` at its default (0).** All results here use it.
- **Describe each character briefly in the prompt.** The references alone separate the two people, but
  writing hair, eyes and clothes per side ("the girl on the left has ... and wears ...") raises fidelity:
  in our cafe test, one girl's outfit changed colour in 3 of 4 seeds without the description and in none
  of 8 with it. This also happens with plain SnapRef and one reference, so it is not caused by the node.
- **Say who is where.** The regions must match the layout of the prompt. A character drawn across a region
  edge loses its reference outside its region (a face on the edge can break).
- **`ref_start`.** We found no clear difference between `0` and `0.1` in reference fidelity. The only
  difference: with `0`, one character was duplicated in 2 of 4 seeds in a scene of two people standing
  close side by side (twice, at two settings); with `0.1` this never happened. `0.1` is the default.
- References can leak onto objects inside a region (e.g. plush toys with the character's face).
- Characters that overlap a lot (hugging, one in front of the other) are not handled.

## Using it with Anima Regional (optional)

It also works with [ComfyUI-Anima-Regional](https://github.com/An1X3R/ComfyUI-Anima-Regional), which can pin
each character's action to a region. Things to know:

- Use exactly the same regions in the Regional layout and in this node. Overlapping regions caused broken faces.
- Apply Regional once per graph.
- **Stale prompt (Anima Regional 0.4.7):** when the model stays loaded and the Regional prompt changes, Regional
  wraps its previous patch and keeps the previous text. Unload the models (*Unload Models* or `POST /free`)
  after changing the prompt. This happens without SnapRef too.
- SnapRef runs a cond/uncond batch as two calls with the original `cond_or_uncond` markers, which made Regional
  route the negative half as well (`unusable cond_or_uncond markers [1, 0] for batch 1`). This node rewrites the
  markers for those calls. The warning still appears once per run while SnapRef encodes the references
  (Regional routes that pass too); that comes from SnapRef and Regional and cannot be fixed here.
