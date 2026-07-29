# Repository guidance

These instructions apply to the entire repository.

## Branch and delivery

- Work only on the branch requested by the user. Do not modify `main`.
- Do not commit or push unless the user explicitly authorizes it.
- Preserve unrelated working-tree changes.

## Privacy invariants

- Face masking is fail closed. When a current segmentation is missing, a
  motion-aligned historical mask may be used only while it remains consistent
  with current tracker position and scale; otherwise full configured geometric
  coverage is immediate. A validated propagated mask may blend toward geometry
  over the configured release window to avoid a hard size flash. No failure may
  silently produce an empty uncovered region.
- Temporal identity is scoped to `(scene_id, track_id)`. Clear tracker,
  segmentation, SAM memory, and temporal history on a scene cut.
- Never transfer identity-owned raw mask history across ambiguous, crossing,
  merged, or reused track IDs. After those guarded per-track states are merged,
  the identity-free final scene mask may be motion-aligned and smoothed across
  Track ID boundaries; it must still reset on cuts and reject disjoint motion.
- `union` adds current tracker geometry to the segmentation contour.
  `intersection` clips the contour to current tracker geometry.
  `mask-only` renders a valid SAM contour without box combination, but
  all failure and drift cases still use geometric coverage.
- Keep the geometric engine on its existing low-overhead single-pass path.
- Preserve atomic output: incomplete encodes must not replace the destination.
  Temporary mask data must be bounded and cleaned on success, failure, or
  cancellation.

## Implementation boundaries

- Temporal segmentation is an offline two-pass pipeline: analyze/store raw
  masks, stabilize within a scene and track, apply the combination policy,
  smooth the merged final scene mask, then render from the source.
- Do not retain every full-resolution mask in RAM. Keep explicit bounds on
  caches, model memory, and temporary storage.
- SAM is optional. Default tests must not require a model download, network,
  CUDA, or real model weights; mock the supported public API contract.
- Keep UI controls and explanations bilingual (English and Chinese), local
  only, and consistent with CLI validation.

## Required validation

Run before handing off code changes:

```text
python -m unittest discover -s tests -v
python -m compileall -q blurface tests scripts
git diff --check
```

The GitHub Actions matrix runs the same unittest discovery and compile check on
Ubuntu and Windows with Python 3.10 and 3.12. New tests under `tests/test_*.py`
are therefore included automatically.
