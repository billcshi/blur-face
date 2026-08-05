# Image and batch-image face masking

## Goal

Version 1.2.0 extends the existing local video anonymization application to
still images. It supports one image or a batch from both the CLI and Local
Studio, while keeping the current video command line and the geometric video
single-pass path unchanged.

Image processing reuses the existing detector, geometric coverage renderer,
blur policies, privacy presets, and optional SAM 2.1 image model. Tracking,
scene detection, temporal stabilization, FFmpeg, and audio settings remain
video-only.

## User-facing contract

### CLI

The existing form remains valid:

```text
blur-face input.mp4 -o output.mp4 --overwrite
```

The positional input accepts one or more paths. An input manifest is also
available for callers that must avoid operating-system command-line limits:

```text
blur-face portrait.jpg -o portrait_blurred.jpg
blur-face first.jpg second.png -o blurred-images/
blur-face photos/ -o blurred-images/
blur-face --input-list private-inputs.json -o blurred-images/
```

- One video file is processed by the unchanged video pipeline.
- One image may target either an explicit supported image filename or an
  output directory. A directory target retains the source filename.
- Multiple image files require an output directory.
- One input directory expands its supported image files non-recursively in a
  deterministic, case-insensitive filename order. Other files are ignored;
  an empty image set is an error.
- Mixing videos, image files, and directories in one command is rejected.
- `--input-list` is mutually exclusive with positional inputs. It contains a
  UTF-8 JSON array of path strings, has a bounded size/count, and is parsed
  without shell expansion. Local Studio always uses this form for multi-image
  jobs so Windows command-line length does not depend on batch size.
- Inputs with the same case-insensitive basename are rejected when they would
  map to the same batch output. This prevents silent replacement on Windows.
- Without `-o`, video retains `output_blur.mp4`, one image uses
  `<stem>_blurred<suffix>`, and an image batch/directory uses a sibling
  `<directory-or-first-stem>_blurred` directory.

Supported input/output formats are JPEG (`.jpg`, `.jpeg`), PNG, WebP, BMP,
and TIFF (`.tif`, `.tiff`). A decoded file must be a non-empty image. Alpha is
preserved for PNG, WebP, and TIFF output; an alpha-bearing input is rejected if
its chosen output format cannot preserve transparency. Detection and masking
operate on a defined BGR view, then the original alpha channel is reattached.
Images are decoded and encoded by OpenCV; EXIF and other source metadata are
not copied to the anonymized output. Removing location/camera metadata is the
privacy-safe default and is documented rather than silently promising
metadata preservation.

Image batches initialize the selected detector and optional SAM model once.
Progress is reported as `completed/total (percent%)`, so the existing Local
Studio status parser can show batch progress.

### Local Studio

The main form gains a bilingual media selector:

- **Video** retains the current single-file picker and output-file picker.
- **Images** uses a native multi-file picker and an output-directory picker.
  The selected paths are shown locally in the form and sent only to the
  loopback service as JSON paths; browser file upload controls are not used.

Image mode hides video-only controls: tracking preset, temporal stabilization
and its dependent settings, SAM correction interval, mask diagnostic *video*
wording, and NVENC. The diagnostic control remains available with image
wording and produces black/blue images using the exact final coverage.

The UI continues to run one job at a time in a child process. For a multi-image
job, the parent writes the JSON input manifest inside its existing private
`job_temp` directory and passes only that manifest path to the child. The
private directory is created inside the actual resolved destination directory
(the output directory for a batch, or the output file's parent for a single
image), not merely beside a possible mount point.
Cancellation stops the process tree and the parent removes the whole private
directory even after a forced kill. The UI holds the destination-parent anchor,
validates the recorded job identity, and refuses symbolic links, junctions, or
replaced paths. A renamed destination parent therefore does not strand the
manifest. Files already completed in a batch remain
valid; the image currently being encoded cannot replace its destination until
its atomic commit succeeds.

## Configuration and dispatch

`AppConfig` remains the validated, single-input video configuration to avoid
changing the large video pipeline. CLI parsing returns either:

- `AppConfig` for one video; or
- a new immutable `ImageBatchConfig` containing the ordered source/output
  pairs plus one validated `AppConfig` that owns all shared masking options.

Media selection combines supported image suffixes, bounded file-signature
inspection, and directory expansion, then confirms the result with the actual
decoder. Known unsupported image signatures (for example GIF, HEIC/AVIF
compatible brands, SVG, and JPEG XL) are rejected
instead of falling into video. A known video signature wins even under a
misleading image suffix, preserving decoder-based video compatibility. Unknown
single-file signatures continue to use the original video pipeline for uncommon
containers; multi-input jobs must be unambiguously supported images.
`ImageBatchConfig.validate()` checks every source, unique output mapping,
source/output separation, supported output suffixes, and all shared
`AppConfig` privacy values before any model is initialized or output created.
Existing destinations are checked for the complete batch up front unless
`--overwrite` is enabled.

The application entry point dispatches `AppConfig` to `VideoProcessor` and
`ImageBatchConfig` to `ImageProcessor`. Heavy OpenCV/model imports stay after
argument validation as they are today.

The UI command builder accepts either its existing `input` string or an
`inputs` array. It uses the same configuration builder as the CLI for path
classification and output mapping. `JobManager` creates its private job
directory inside the actual output directory/filesystem before launching the
child; for multiple inputs it writes a bounded JSON manifest there and invokes
the authoritative CLI with `--input-list <manifest>`. The parent holds the
resolved destination directory open and creates/removes the job relative to
that anchor, so cleanup does not depend on the original path remaining named.
No shell interpolation is used.

## Image pipeline

For a batch, `ImageProcessor.run()` performs these steps:

1. Validate every source/output mapping and destination conflict.
2. Initialize `FaceDetector` once. Initialize `Sam2Segmenter` once only when
   SAM is selected and debug mode is off.
3. For each source, decode an oriented BGR or BGRA image, retain alpha when
   present, and run the detector once on the BGR pixels at the configured
   threshold.
4. Reject detector boxes below `min_face_size`, above
   `max_face_height_ratio`, non-finite, or outside a valid clipped area.
5. Keep an untouched image copy as the SAM source. For every valid detection,
   compute the same expanded ROI and inner detector box used by the renderer.
6. In geometric mode, invoke the existing `apply_blur` or
   `apply_mask_preview` path with the configured shape, scale, and
   confidence-derived kernel. Kernel settings are a 1080p short-edge baseline;
   the shared image/video calculation scales them up for higher-resolution
   media so still images do not receive perceptually weaker anonymization.
7. In SAM mode, prompt `Sam2Segmenter` with the original ROI through a new
   `build_contour()` method. It returns only a cleaned contour; the existing
   renderer centrally applies `union`, `intersection`, or `mask-only`, matching
   video semantics. The existing public `build_mask()` API and its
   union/intersection contract remain unchanged for compatibility and delegate
   to the contour primitive before applying their current policy.
8. Validate each SAM contour with the existing coverage/coherence/drift gate.
   Missing, empty, sparse, thin, fragmented, non-finite, or implausibly
   displaced masks immediately use full configured geometric coverage in all
   combination modes. A model exception is logged with a bounded warning
   count and cannot create an uncovered face region.
9. Resolve and create the actual destination directory, record its filesystem
   identity, freeze every source/output mapping under that canonical directory,
   then create the private job directory inside it (batch: inside the output directory; single image:
   inside the output file's parent). Encode the finished pixels to a temporary
   file in that private directory, verify non-empty encoded bytes, and flush
   them. This remains on the destination filesystem even when the destination
   directory itself is a mount point. With
   `--overwrite`, `os.replace()` commits the file. Without it, an atomic
   create-if-absent commit (a same-filesystem hard link followed by unlinking
   the temporary name) fails if the destination appeared after preflight;
   there is no check-then-replace race. On POSIX, the output and job directories
   remain open and create/link/replace/unlink operations are relative to those
   descriptors. On Windows, non-delete-sharing directory handles prevent either
   directory from being renamed or replaced during path-based commits. Thus
   replacing the directory or retargeting the originally selected symlink
   cannot redirect an output. If the filesystem does not support the
   no-overwrite hard-link commit, the operation fails safely and leaves the
   destination untouched; it never falls back to replacement. On a normal
   exception or cancellation the child removes its exact temporary file.
   After UI forced termination, the parent removes the UI-owned job directory.

Rendering multiple overlapping faces is safe because detection and SAM always
read original pixels, while masks are applied to a separate output image.
In `--debug` mode SAM is intentionally not initialized, no pixels are blurred,
and each accepted detection is drawn with the existing confidence-labelled
debug box, matching video debug behavior.

## Safety and resource bounds

- Image processing is local after optional model initialization, like video.
- The batch keeps at most the current original image, current rendered image,
  one current ROI/mask, and model state in memory. It never retains decoded
  images or masks for the whole batch.
- Geometry remains the fallback for every SAM failure or validity rejection.
- Image jobs have no identity history, so no mask or prompt crosses images.
- Alpha-bearing images retain their alpha channel while the underlying color
  pixels are anonymized. Formats that cannot retain alpha fail without output.
- Each output is atomic. Existing output files are untouched until their
  corresponding replacement is completely encoded. A no-overwrite commit
  also atomically rejects a destination created during a long-running batch.
- Temporary files and the Local Studio input manifest use unique names inside
  the exact job-owned directory on the output filesystem. The child cleans it
  on ordinary exit; the UI parent also cleans it after graceful or forced
  cancellation.
- Batch validation happens before processing to catch collisions and existing
  destinations. A later decode/model failure stops the job; already committed
  outputs are reported and remain usable, while the failing image has no
  partial destination.

## Tests and acceptance criteria

New default tests use synthetic images and mocked public detector/SAM APIs;
they require no network, model download, CUDA, or real model weight.

- CLI classification/default-output tests cover video compatibility, one
  image, multiple images, a directory, mixed-media rejection, empty
  directories, duplicate output names, unsupported suffixes, bounded JSON
  manifests, and positional/manifest mutual exclusion.
- Image integration tests cover geometric blur, image mask preview, model reuse
  across a batch, SAM `union`/`intersection`/`mask-only`, invalid-SAM geometric
  fallback, non-finite/missing detector confidence, EXIF-oriented JPEG decode,
  every documented codec, alpha retention, metadata removal, encoding failure,
  atomic success/failure, destination appearance during no-overwrite commit,
  and output-directory symlink retargeting under overwrite.
- UI tests cover multi-image command construction, manifest placement, native
  picker responses, bilingual image-mode help mappings, private-temporary
  cleanup, symlink/replaced-directory refusal, and existing token/cancellation
  behavior.
- Batch-failure tests confirm already committed images remain valid, the
  currently failing destination is absent/unchanged, and status/logs identify
  the completed count and failing input.
- Documentation and version metadata identify 1.2 image support and clearly
  state supported formats, metadata removal, batch mapping, atomicity, and
  the absence of automatic-detection guarantees.
- The repository-required unittest discovery, compile check, and diff check
  pass on the supported Python matrix without optional dependencies.
