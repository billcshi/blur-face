# Offline temporal face-mask stabilization

## Decision

Segmentation engines use a two-pass input pipeline when temporal stabilization
is enabled. The current stream immediately mutates and encodes frame `N-k`, so
it cannot safely repair that frame after a face is first detected at frame
`N`. A fixed output delay would only solve a bounded special case and would
still make scene-wide forward/backward correction and atomic failure handling
harder.

Pass 1 analyzes original frames and writes only bounded, compressed working
data to an automatically removed temporary directory:

- tracker observations and geometric coverage boxes in SQLite;
- color motion/appearance proxies whose longest side is at most 480 pixels;
- proxy-resolution full-frame PNG masks per `(scene, track ID, frame)`, never permanent
  full-resolution mask arrays in RAM.

After pass 1, the temporal stage repairs masks within each scene. Pass 2 opens
the original video again, reads the repaired records, applies the selected
blur, and streams frames to the existing atomic encoder. A failed analysis,
temporary-store limit, render, or encode leaves no partial output committed.
Temporary storage has a hard 4 GiB default ceiling and is removed on success,
failure, and cancellation.

SAM reverse propagation additionally keeps at most 16 original frames
(`15` backfill frames) and never more than 512 MiB of raw frames in RAM; on
very high-resolution video the byte ceiling shortens the effective reverse
window.

The geometric engine keeps its existing single-pass behavior and cost.

## Identity and scene boundaries

Temporal state is keyed by `(scene ID, tracker Track ID)`. A scene-cut detector
uses downscaled grayscale histogram distance plus pixel-change evidence. A cut
increments the scene ID, resets the tracker, clears SAM memory and all
in-memory temporal state, and prevents queries or propagation across the cut.

Backfill starts only from a newly reliable, non-edge observation. It walks at
most `backfill_frames` earlier frames and stops on:

- a scene boundary;
- failed or inconsistent forward/backward optical flow;
- excessive displacement, scale/area change, or overlap with another track;
- evidence that the face truly entered through the frame boundary;
- weak appearance agreement.

Every accepted mask remains owned by the same `(scene, track ID)`. There is no
nearest-mask reassignment. Crossing/overlap ambiguity stops propagation rather
than transferring history. Ambiguous IDs are not written back into SAM memory;
they use geometry until they separate, receive a new high-confidence
observation, and pass the mask/box drift gate. Re-entry after removal receives
fresh state.

## Stabilization and privacy semantics

Each track's raw segmentation contour is stabilized before tracks are merged.
Masks are first motion-aligned with optical flow (SAM uses its video memory
where available), then fused with asymmetric hysteresis:

- strong newly present pixels enter immediately;
- a later, more complete contour may repair an earlier contained
  under-segmentation within `backfill_frames`;
- aligned pixels that temporarily disappear are held for the configured
  `release_hold_frames`;
- contained shape/area changes are blended, while spatially inconsistent
  motion is applied immediately to prevent trails;
- on an intermittent segmentation failure, a motion-aligned historical mask
  may continue only while its center, overlap, and scale remain consistent
  with the current tracker; it blends toward geometry over
  `release_hold_frames`, while an inconsistent or unavailable propagation
  immediately uses full configured geometry;
- all history is cancelled on a scene cut or ambiguous identity crossing.

Detector/SAM correction frames are treated as observations in the same aligned
state, rather than hard replacements.

The combination policy is applied **after per-track stabilization**. This
keeps detector-box jitter out of the identity-owned temporal contour:

- `union`: stable contour plus current geometric tracker coverage;
- `intersection`: stable contour clipped to current geometric coverage.
- `mask-only`: the stable SAM contour is rendered unchanged. The selected detector
  remains responsible for discovery, tracking, prompts, and correction, but
  does not define the successful segmentation contour.

Before combination, a shared validity gate checks actual mask/box pixel
coverage, foreground area, dominant-component coherence, face-core contact,
two-dimensional horizontal/vertical support, centroid displacement, and
excessive growth. A successful contour must cover at least half of the current
face proposal and span both axes; centered eye strips, vertical slivers, and
sparse corner components therefore cannot masquerade as a face. Missing,
non-finite, or low object scores and propagation, decoding, or optical-flow
failures are marked for geometric fallback. The same gate is applied again
after per-track stabilization, before combination. Fallback is rendered
regardless of the combination mode.

After the policy is applied, all included Track IDs and fallback regions are
merged into one identity-free mask for each frame. A second bounded temporal
stage motion-aligns these final masks within the same scene and interpolates
their signed-distance contours over `backfill_frames` in reverse and
`release_hold_frames` forward. This is what smooths a face even when its Track
ID is rebuilt or the winning source changes between SAM and geometry.
If dense flow is unavailable or rejected, this stage may use the existing
image coordinates only when the two masks already have strong overlap, similar
area, small centroid displacement, and no distant unmatched lobe. Otherwise it
keeps the current mask unchanged.
Components are associated independently: a newly appearing face is admitted
immediately, and one stable face cannot authorize interpolation of another
disjoint face. Current-frame coverage is never removed: reverse smoothing
distributes validated growth into earlier frames, while forward smoothing
controls release/shrinkage. Disjoint rapid motion is accepted immediately
instead of blended, and a scene cut always resets the final state. The raw
per-track histories remain isolated, so the final union cannot seed SAM memory
or reassign an identity.

The optional mask-preview output changes only the final pixel effect: the
ordinary blur and the preview both consume the same post-stabilization,
post-combination coverage. Preview frames start black and paint that exact
coverage blue, including geometric fallback. They retain the source
dimensions, frame rate, and frame count but omit source pixels and audio, so
the result can be measured without exposing the original content.

SAM video memory is the primary source of object propagation in SAM mode.
Optical flow is still used as a bounded, model-independent alignment signal for
hysteresis, correction-seam comparison, reverse-propagation validation, and
fail-closed fallback when the installed SAM API cannot reverse a window. It is
not used to replace healthy SAM memory or to reassign a mask between tracks.

## SAM 2

SAM 2 uses one video session per scene. Detector boxes seed newly discovered
objects and periodically correct existing objects without resetting unrelated
object memory. To prevent unbounded model history on long uncut footage, the
streaming session is re-seeded from current tracker coverage after at most 30
streamed frames or 16 object memories. Retained session tensors are measured
and the session is immediately discarded for re-seeding if they exceed
512 MiB; disk-backed temporal fusion bridges that
bounded correction seam. Forward propagation uses the streamed session. Backfill uses
the official video-session reverse direction over the bounded cached scene
window when available; optical-flow backfill remains the fail-closed,
version-compatible fallback. API failures reset only the current scene's SAM
state and emit geometric fallback records.

If a frame has more than 16 active faces, SAM is skipped for that frame/window
and every face uses geometry instead of risking an out-of-memory failure. A
single raw frame larger than the 512 MiB reverse-cache ceiling is never cached;
the disk-backed optical-flow repair remains available.

## User controls

The temporal switch is shown only for `sam2.1` and defaults on. The exposed
controls are:

- backfill frames (default 10);
- release hold frames (default 5);
- scene-cut sensitivity (default 0.55);
- temporary storage limit (default 4096 MiB).

The UI explains in English and Chinese that stabilization is offline, slower,
and uses bounded temporary storage.
