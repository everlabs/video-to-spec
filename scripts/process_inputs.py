#!/usr/bin/env python3
"""Extract deduplicated frames and a frame-matched transcript timeline from a
screen-recording video + SRT pair.

Usage:
    process_inputs.py <video> <srt> <output-dir> [--threshold N] [--backend B]

Backends (auto-detected, in preference order):
    pixel         mean absolute pixel difference on 64x64 grayscale
                  thumbnails that ffmpeg produces in one extra pass.
                  No dependencies beyond ffmpeg; compares in milliseconds.
    imagemagick   `magick compare -metric PHASH` (ImageMagick 7)
    imagemagick6  `compare -metric PHASH`        (ImageMagick 6 legacy)
    imagehash     Python `imagehash` + `Pillow`

`pixel` is the default because the perceptual backends are wrong for this job
in two ways. Correctness: PHASH is *designed* to be robust to small visual
changes, but on a screen recording the changes that matter - a modal opening,
a field filling in, an error appearing - are perceptually small and
semantically large, so a threshold loose enough to ignore cursor drift also
ignores them (observed: a 6.5-minute portrait recording collapsing to a single
frame). Speed: both ImageMagick backends spawn one `compare` process per frame
pair, which costs minutes on a recording of any length.

Per-backend default thresholds (tuned for screen recordings: ignore mouse-
cursor drift, catch real UI changes like modals, scrolls, form input):
    pixel                       → 0.1 (percent of pixels changed)
    imagemagick / imagemagick6  → 200
    imagehash                   → 5 (hamming distance out of 64)

Raise --threshold if near-identical frames survive; lower it if real screen
states are being merged away.

Produces (under <output-dir>/_work/):
    frames-all/HH-MM-SS.png   one image per unique screen state
    timeline.md               every SRT segment with the frame that was most
                              recently on screen (looking up to 1s ahead per
                              spec, so a frame change inside the spoken second
                              still attaches)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "pixel": 0.1,
    "imagemagick": 200.0,
    "imagemagick6": 200.0,
    "imagehash": 5.0,
}

# `pixel` backend tuning, measured on a synthetic 8-screen portrait screencast
# (40 frames, cursor drift within each screen), clean and with heavy encode
# noise. At 128x128 / delta 32 the smallest *real* change (a form field filling
# in, a validation line appearing, a status dot) moved 0.22% of pixels while
# the loudest cursor drift moved 0.067% - a 3.3x margin that did not shift when
# the same content was re-encoded at 30fps with added sensor noise, because the
# downscale averages pixel noise away. The default threshold sits between them.
SIG_EDGE = 128
# A pixel counts as changed only past this 0-255 delta, which ignores
# antialiasing and compression ringing while catching real repaints.
PIXEL_DELTA = 32


def fail(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def detect_backend(preferred: str | None = None) -> tuple[str, list[str] | None]:
    """Return (backend_name, magick_cmd_prefix_or_None).

    Tries ImageMagick first (already on most dev machines as a general image
    tool), then falls back to a Python imagehash install.
    """
    candidates: list[str]
    if preferred:
        candidates = [preferred]
    else:
        candidates = ["pixel", "imagemagick", "imagemagick6", "imagehash"]

    for c in candidates:
        if c == "pixel":
            # ffmpeg is already a hard requirement, so this always works.
            return ("pixel", None)
        if c == "imagemagick" and shutil.which("magick"):
            return ("imagemagick", ["magick", "compare", "-metric", "PHASH"])
        if c == "imagemagick6" and shutil.which("compare"):
            # ImageMagick 6 ships `compare` as a separate binary.
            return ("imagemagick6", ["compare", "-metric", "PHASH"])
        if c == "imagehash":
            try:
                import imagehash  # noqa: F401
                from PIL import Image  # noqa: F401
            except ImportError:
                continue
            return ("imagehash", None)

    fail(
        "No image comparison backend available. Install one of:\n"
        "  ImageMagick (preferred):  brew install imagemagick\n"
        "  Python imagehash:         pip3 install --user imagehash Pillow"
    )
    return ("", None)  # unreachable; keeps type checkers happy


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg not found. Install with: brew install ffmpeg")


def seconds_to_ts(s: int) -> str:
    return f"{s // 3600:02d}-{(s % 3600) // 60:02d}-{s % 60:02d}"


def extract_frames(video: Path, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    # fps=1 → frame_000001.png is t=0s, frame_000002.png is t=1s, ...
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        "fps=1",
        "-q:v",
        "2",
        str(raw_dir / "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(raw_dir.glob("frame_*.png"))


def extract_signatures(video: Path, work_dir: Path) -> list[bytes]:
    """One extra ffmpeg pass emitting a 64x64 grayscale thumbnail per second.

    Written as a single raw stream so the whole recording costs one subprocess
    and SIG_EDGE**2 bytes per frame, rather than a process per comparison.
    """
    raw = work_dir / "signatures.gray"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps=1,scale={SIG_EDGE}:{SIG_EDGE},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", str(raw),
    ]
    subprocess.run(cmd, check=True)
    blob = raw.read_bytes()
    raw.unlink(missing_ok=True)
    size = SIG_EDGE * SIG_EDGE
    return [blob[i:i + size] for i in range(0, len(blob) - size + 1, size)]


def _changed_fraction(a: bytes, b: bytes) -> float:
    """Percentage of pixels that changed by more than PIXEL_DELTA.

    Counting *how many* pixels changed beats averaging *how much* they changed:
    UI edits are localised, so a filled form field alters a small region a lot.
    A mean over the whole frame dilutes that below the cursor-drift floor,
    while a changed-pixel count separates them cleanly.
    """
    if len(a) != len(b) or not a:
        return 100.0
    changed = sum(1 for x, y in zip(a, b) if (x - y if x > y else y - x) > PIXEL_DELTA)
    return 100.0 * changed / len(a)


def dedupe_pixel(
    frames: list[Path], signatures: list[bytes], threshold: float
) -> list[tuple[int, Path]]:
    """Keep a frame when it differs enough from the last *kept* frame.

    Comparing against the last kept frame rather than the previous one means a
    slow scroll accumulates until it crosses the threshold, instead of every
    step being individually too small to register.
    """
    kept: list[tuple[int, Path]] = []
    last_sig: bytes | None = None
    for f in frames:
        m = re.search(r"frame_(\d+)\.png$", f.name)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if idx >= len(signatures):
            # ffmpeg's two passes disagreed on frame count; keep the frame
            # rather than silently dropping a screen state.
            kept.append((idx, f))
            continue
        sig = signatures[idx]
        if last_sig is None or _changed_fraction(last_sig, sig) > threshold:
            kept.append((idx, f))
            last_sig = sig
    return kept


def _phash_magick(cmd: list[str], a: Path, b: Path) -> float:
    """Return PHASH distance between two images via `magick compare`.

    Output format on stderr/stdout looks like:
        12345.67 (0.0123)
    where the first number is the raw PHASH metric. `compare` returns
    exit code 1 when images differ, which we treat as success.
    """
    result = subprocess.run(
        [*cmd, str(a), str(b), "null:"],
        capture_output=True,
        text=True,
    )
    # ImageMagick writes the metric to stderr (in v7) or stdout (in v6).
    blob = (result.stderr or result.stdout).strip()
    if not blob:
        return 0.0
    # First whitespace-separated token is the raw metric.
    first = blob.split()[0]
    try:
        return float(first)
    except ValueError:
        return 0.0


def dedupe_magick(
    frames: list[Path], cmd: list[str], threshold: float
) -> list[tuple[int, Path]]:
    kept: list[tuple[int, Path]] = []
    if not frames:
        return kept
    # First frame always kept.
    m0 = re.search(r"frame_(\d+)\.png$", frames[0].name)
    if m0:
        kept.append((int(m0.group(1)) - 1, frames[0]))
    last_kept = frames[0]
    for f in frames[1:]:
        m = re.search(r"frame_(\d+)\.png$", f.name)
        if not m:
            continue
        second = int(m.group(1)) - 1
        d = _phash_magick(cmd, last_kept, f)
        if d > threshold:
            kept.append((second, f))
            last_kept = f
    return kept


def dedupe_imagehash(
    frames: list[Path], threshold: float
) -> list[tuple[int, Path]]:
    from PIL import Image
    import imagehash

    kept: list[tuple[int, Path]] = []
    prev_hash = None
    for f in frames:
        m = re.search(r"frame_(\d+)\.png$", f.name)
        if not m:
            continue
        second = int(m.group(1)) - 1
        h = imagehash.phash(Image.open(f))
        if prev_hash is None or (h - prev_hash) > threshold:
            kept.append((second, f))
            prev_hash = h
    return kept


def parse_srt(srt_path: Path) -> list[dict]:
    raw = srt_path.read_text(encoding="utf-8", errors="replace")
    raw = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    segments = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        ts_line_idx = 1 if pattern.search(lines[1] if len(lines) > 1 else "") else 0
        m = pattern.search(lines[ts_line_idx])
        if not m:
            continue
        h1, m1, s1, _ms1, h2, m2, s2, _ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1
        end = h2 * 3600 + m2 * 60 + s2
        text = " ".join(lines[ts_line_idx + 1 :]).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def attach_frames(
    segments: list[dict], kept: list[tuple[int, Path]]
) -> list[dict]:
    """For each segment, attach the most recent kept frame ≤ start+1s."""
    out = []
    for seg in segments:
        cutoff = seg["start"] + 1
        best: tuple[int, Path] | None = None
        for (sec, path) in kept:
            if sec <= cutoff:
                best = (sec, path)
            else:
                break
        out.append(
            {
                **seg,
                "frame_second": best[0] if best else None,
                "frame_path": best[1] if best else None,
            }
        )
    return out


def write_outputs(
    timeline: list[dict],
    kept: list[tuple[int, Path]],
    work_dir: Path,
) -> None:
    frames_all = work_dir / "frames-all"
    frames_all.mkdir(parents=True, exist_ok=True)
    for (sec, src) in kept:
        dst = frames_all / f"{seconds_to_ts(sec)}.png"
        if not dst.exists():
            shutil.copy2(src, dst)
    lines = ["# Timeline", ""]
    lines.append(
        "Each block is one transcript segment with the screen state that was "
        "most recently visible (allowing up to 1s lookahead). Frame paths are "
        "relative to this directory."
    )
    lines.append("")
    for seg in timeline:
        ts_start = seconds_to_ts(seg["start"])
        ts_end = seconds_to_ts(seg["end"])
        lines.append(f"## {ts_start} → {ts_end}")
        if seg["frame_second"] is not None:
            frame_ts = seconds_to_ts(seg["frame_second"])
            lines.append(f"frame: `frames-all/{frame_ts}.png`")
        else:
            lines.append("frame: _(none yet)_")
        lines.append("")
        lines.append(seg["text"])
        lines.append("")
    (work_dir / "timeline.md").write_text("\n".join(lines), encoding="utf-8")
    raw_dir = work_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="Path to source video")
    ap.add_argument("srt", help="Path to SRT transcript")
    ap.add_argument("output_dir", help="Output directory (e.g. docs/video-to-spec/<spec>)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Dedup threshold (overrides per-backend default).",
    )
    ap.add_argument(
        "--backend",
        choices=["pixel", "imagemagick", "imagemagick6", "imagehash"],
        default=None,
        help="Force a specific backend (default: pixel).",
    )
    args = ap.parse_args()

    check_ffmpeg()
    backend, magick_cmd = detect_backend(args.backend)
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS[backend]
    print(f"backend: {backend}  threshold: {threshold}", file=sys.stderr)

    video = Path(args.video).expanduser().resolve()
    srt = Path(args.srt).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not video.is_file():
        fail(f"video not found: {video}")
    if not srt.is_file():
        fail(f"transcript not found: {srt}")

    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "raw"

    print(f"[1/4] Extracting 1fps frames from {video.name}", file=sys.stderr)
    frames = extract_frames(video, raw_dir)
    print(f"      → {len(frames)} raw frames", file=sys.stderr)

    print(f"[2/4] Deduplicating via {backend}", file=sys.stderr)
    if backend == "pixel":
        signatures = extract_signatures(video, work_dir)
        kept = dedupe_pixel(frames, signatures, threshold)
    elif backend in ("imagemagick", "imagemagick6"):
        kept = dedupe_magick(frames, magick_cmd, threshold)
    else:
        kept = dedupe_imagehash(frames, threshold)
    print(f"      → {len(kept)} unique screen states", file=sys.stderr)
    if frames and len(kept) <= 1 < len(frames):
        print(
            "      warning: everything collapsed to one screen state - the "
            f"threshold ({threshold}) is likely too high for this recording. "
            "Re-run with a lower --threshold.",
            file=sys.stderr,
        )

    print(f"[3/4] Parsing SRT {srt.name}", file=sys.stderr)
    segments = parse_srt(srt)
    print(f"      → {len(segments)} transcript segments", file=sys.stderr)

    print("[4/4] Matching frames to segments and writing timeline", file=sys.stderr)
    timeline = attach_frames(segments, kept)
    write_outputs(timeline, kept, work_dir)

    print("", file=sys.stderr)
    print(f"Done. Read: {work_dir}/timeline.md", file=sys.stderr)
    print(f"Frames in: {work_dir}/frames-all/", file=sys.stderr)


if __name__ == "__main__":
    main()
