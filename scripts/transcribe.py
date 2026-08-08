#!/usr/bin/env python3
"""Generate an SRT transcript for a video (or audio) file.

Providers, auto-detected in preference order (local tools first):
    mlx-whisper     `mlx_whisper` CLI - local, free, fast on Apple Silicon
    openai-whisper  `whisper` CLI - local, free
    openai-api      OpenAI transcription API. Key from $OPENAI_API_KEY, else
                    the macOS keychain (generic password with service name
                    OPENAI_API_KEY).
    command         any user-supplied template, e.g.
                    --provider command --command 'mytool {audio} -o {srt}'
                    Placeholders {video} {audio} {srt} are substituted
                    shell-quoted. If {audio} appears, a 16 kHz mono mp3 is
                    extracted first.

Quality tiers (--quality, per provider):
    fast (default)  mlx-whisper    mlx-community/whisper-large-v3-turbo
                    openai-whisper turbo
                    openai-api     gpt-4o-transcribe-diarize
    max             mlx-whisper    mlx-community/whisper-large-v3-mlx
                    openai-whisper large-v3
                    openai-api     gpt-4o-transcribe-diarize (same; no
                                   higher-accuracy timestamped model exists)
`--model` overrides the tier entirely.

This skill needs *timestamps* - every cue is matched to a video frame - which
constrains the API model choice. Only two OpenAI models emit them:
    gpt-4o-transcribe-diarize  segment start/end via response_format
                               diarized_json (converted to SRT here).
                               Requires chunking_strategy, always.
    whisper-1                  native SRT, but markedly weaker on non-English
                               (it Russifies Ukrainian). Legacy fallback only.
`gpt-transcribe` is OpenAI's most accurate transcription model but returns
JSON/text with no timestamps at all, so it cannot drive this pipeline; asking
for it fails fast with an explanation rather than silently degrading.

Usage:
    transcribe.py --detect
    transcribe.py <video> [--srt-out PATH] [--provider P]
                  [--quality fast|max] [--model M] [--language xx]
                  [--command TEMPLATE] [--chunk-seconds N]

The API path chunks long audio (default 900 s per chunk, ~3.5 MB at 32 kbps,
safely under the 25 MB upload limit), transcribes each chunk, offsets the
timestamps by the real duration of the preceding chunks, and merges into one
renumbered SRT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.openai.com/v1/audio/transcriptions"
TS_PATTERN = re.compile(
    r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})"
)

# Per-provider model for each quality tier. --model overrides both.
MODELS = {
    "mlx-whisper": {
        "fast": "mlx-community/whisper-large-v3-turbo",
        "max": "mlx-community/whisper-large-v3-mlx",
    },
    "openai-whisper": {"fast": "turbo", "max": "large-v3"},
    # Both tiers are the same model: it is the only current-generation OpenAI
    # model that returns timestamps at all.
    "openai-api": {
        "fast": "gpt-4o-transcribe-diarize",
        "max": "gpt-4o-transcribe-diarize",
    },
}

# API models that cannot drive this pipeline, and why.
UNTIMESTAMPED_API_MODELS = {
    "gpt-transcribe": "OpenAI's most accurate transcription model, but it "
                      "returns plain JSON/text with no segment timestamps",
    "gpt-4o-transcribe": "returns plain JSON/text with no segment timestamps",
    "gpt-4o-mini-transcribe": "returns plain JSON/text with no segment "
                              "timestamps",
}

# Model names that route to the API rather than a local whisper CLI, so that a
# bare `--model whisper-1` is not handed to mlx_whisper as a HuggingFace repo.
API_MODEL_RE = re.compile(r"^(whisper-1|gpt-[\w.-]*transcribe[\w.-]*)$")
SNAPSHOT_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def provider_for_model(model: str) -> str | None:
    """Infer the provider a `--model` belongs to, or None if ambiguous.

    Bare whisper names (`turbo`, `large-v3`) fit either local CLI, so they stay
    ambiguous and fall through to normal auto-detection.
    """
    if API_MODEL_RE.match(model):
        return "openai-api"
    if model.startswith("mlx-community/") or model.startswith("mlx_"):
        return "mlx-whisper"
    return None


def untimestamped_reason(model: str) -> str | None:
    """Why `model` cannot drive this pipeline, or None if it can.

    Only two OpenAI models emit timestamps: whisper-1 (native SRT) and the
    diarize family (segment start/end). Everything else is checked against the
    denylist with any dated snapshot suffix stripped, so pinned names like
    `gpt-4o-mini-transcribe-2025-06-03` are rejected before the upload rather
    than by a 400 after it.
    """
    if model == "whisper-1" or "-diarize" in model:
        return None
    base = SNAPSHOT_SUFFIX_RE.sub("", model)
    return UNTIMESTAMPED_API_MODELS.get(base)


def fail(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# --- detection ---------------------------------------------------------------


def openai_key() -> tuple[str, str] | None:
    """Return (key, source) or None. Never print the key itself."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return (key, "env")
    if sys.platform == "darwin" and shutil.which("security"):
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "OPENAI_API_KEY", "-w"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return (result.stdout.strip(), "macOS keychain")
    return None


def _runnable(candidate: Path) -> bool:
    """Is this console script actually executable *and* launchable?

    `shutil.which` implies runnable; a venv path does not. A venv whose
    interpreter was removed (a pyenv upgrade, a deleted base Python) leaves the
    entry point executable but its shebang dangling, and subprocess.run would
    die with a bare FileNotFoundError. Check the shebang so a stale venv is
    reported as "not found" and detection falls through to the next provider.
    """
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return False
    try:
        with candidate.open("rb") as fh:
            if fh.read(2) != b"#!":
                return True  # binary (e.g. Windows .exe) - nothing to verify
            interpreter = fh.readline().decode("utf-8", "replace").strip()
    except OSError:
        return False
    if not interpreter:
        return False
    # Handle `#!/usr/bin/env python3` as well as a direct interpreter path.
    parts = interpreter.split()
    exe = parts[0]
    if exe.endswith("env") and len(parts) > 1:
        return shutil.which(parts[1]) is not None
    return Path(exe).exists()


def find_tool(name: str) -> str | None:
    """Locate a CLI on PATH, or in a virtualenv alongside the skill.

    Installing mlx-whisper into `<skill>/.venv` is the tidiest way to keep a
    multi-gigabyte ML dependency out of the system Python, but it leaves the
    CLI off PATH unless the venv is activated. Look there too so the skill
    works straight after `python3 -m venv .venv && .venv/bin/pip install ...`.
    """
    found = shutil.which(name)
    if found:
        return found
    skill_root = Path(__file__).resolve().parent.parent
    # POSIX venvs put console scripts in bin/; Windows uses Scripts/ with an
    # executable suffix.
    if os.name == "nt":
        subdirs, suffixes = ("Scripts", "bin"), (".exe", ".bat", "")
    else:
        subdirs, suffixes = ("bin",), ("",)
    for venv in (skill_root / ".venv", skill_root / "venv"):
        for subdir in subdirs:
            for suffix in suffixes:
                candidate = venv / subdir / (name + suffix)
                if _runnable(candidate):
                    return str(candidate)
    return None


def detect_providers() -> list[str]:
    available = []
    if find_tool("mlx_whisper"):
        available.append("mlx-whisper")
    if find_tool("whisper"):
        available.append("openai-whisper")
    if openai_key():
        available.append("openai-api")
    return available


def print_detect_report() -> None:
    key = openai_key()
    for name, tool in (("mlx-whisper", "mlx_whisper"), ("openai-whisper", "whisper")):
        path = find_tool(tool)
        if path and not shutil.which(tool):
            print(f"{name}: available (skill venv: {path})")
        else:
            print(f"{name}: {'available' if path else 'not found'}")
    print(f"openai-api: {f'available (key in {key[1]})' if key else 'not found'}")
    for extra in ("whisper-cli", "whisper-cpp"):
        if shutil.which(extra):
            print(f"note: `{extra}` is on PATH - usable via --provider command")
    usable = detect_providers()
    print(f"usable: {', '.join(usable) if usable else 'none'}")
    if usable:
        chosen = usable[0]
        print(
            f"default: {chosen} / {MODELS[chosen]['fast']} "
            f"(--quality max -> {MODELS[chosen]['max']})"
        )


# --- media helpers -----------------------------------------------------------


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        fail("ffmpeg/ffprobe not found. Install with: brew install ffmpeg")


def media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_audio(video: Path, workdir: Path) -> Path:
    out = workdir / "audio.mp3"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "32k", str(out),
        ],
        check=True,
    )
    return out


def segment_audio(audio: Path, workdir: Path, chunk_seconds: int) -> list[Path]:
    seg_dir = workdir / "chunks"
    seg_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio), "-f", "segment",
            "-segment_time", str(chunk_seconds), "-c", "copy",
            str(seg_dir / "chunk_%04d.mp3"),
        ],
        check=True,
    )
    return sorted(seg_dir.glob("chunk_*.mp3"))


# --- SRT parse / write -------------------------------------------------------


def parse_srt_ms(text: str) -> list[tuple[int, int, str]]:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        ts_idx = next((i for i, ln in enumerate(lines) if TS_PATTERN.search(ln)), None)
        if ts_idx is None:
            continue
        m = TS_PATTERN.search(lines[ts_idx])
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = ((h1 * 60 + m1) * 60 + s1) * 1000 + ms1
        end = ((h2 * 60 + m2) * 60 + s2) * 1000 + ms2
        body = " ".join(lines[ts_idx + 1:]).strip()
        if body:
            cues.append((start, end, body))
    return cues


def fmt_ts(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[tuple[int, int, str]], path: Path) -> None:
    lines = []
    for i, (start, end, body) in enumerate(cues, 1):
        lines += [str(i), f"{fmt_ts(start)} --> {fmt_ts(end)}", body, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --- providers ---------------------------------------------------------------


def _multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = "----video-to-spec-" + os.urandom(12).hex()
    parts = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{file_path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + file_path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _openai_call(audio: Path, key: str, model: str, language: str | None) -> str:
    diarize = "-diarize" in model
    fields = {
        "model": model,
        "response_format": "diarized_json" if diarize else "srt",
    }
    if diarize:
        # Rejected outright without this, at any audio length:
        # "chunking_strategy is required for diarization models".
        fields["chunking_strategy"] = "auto"
    if language:
        fields["language"] = language
    body, content_type = _multipart(fields, audio)
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        fail(f"OpenAI API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        fail(f"network error calling OpenAI API: {e.reason}")
    except OSError as e:
        # A socket read timeout raises TimeoutError, which is an OSError but
        # *not* a URLError - without this the caller dies on a raw traceback
        # and every already-billed chunk is discarded.
        fail(f"network error calling OpenAI API: {e}")
    return ""  # unreachable


def parse_diarized_json(text: str) -> list[tuple[int, int, str]]:
    """Convert a diarized_json response into (start_ms, end_ms, text) cues.

    Shape (verified against the live API):
        {"text": ..., "duration": 37.08, "usage": {...},
         "segments": [{"type": "transcript.text.segment", "id": "seg_0",
                       "speaker": "A", "text": " ...",
                       "start": 0.0, "end": 2.15}, ...]}
    Speaker labels are dropped: these recordings are near-always one narrator,
    and a "A:" prefix on every cue would just be noise downstream.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        fail(f"could not parse diarized_json response: {e}")
        return []  # unreachable; fail() exits
    if not isinstance(payload, dict):
        fail(
            "unexpected diarized_json response: expected a JSON object, got "
            f"{type(payload).__name__}"
        )
    segments = payload.get("segments")
    if not isinstance(segments, list):
        fail("diarized_json response has no `segments` array")
        return []
    cues = []
    dropped = 0
    for seg in segments:
        if not isinstance(seg, dict):
            dropped += 1
            continue
        body = (seg.get("text") or "").strip()
        start, end = seg.get("start"), seg.get("end")
        if start is None or end is None:
            # Losing spoken text silently is the one failure this skill cannot
            # afford - the whole point is that no instruction goes missing.
            if body:
                dropped += 1
            continue
        if not body:
            continue
        try:
            cues.append((round(float(start) * 1000), round(float(end) * 1000), body))
        except (TypeError, ValueError):
            dropped += 1
    if dropped:
        print(
            f"warning: dropped {dropped} segment(s) with missing or unusable "
            "timestamps - transcript may be incomplete",
            file=sys.stderr,
        )
    # diarized output is per-speaker and not guaranteed chronological, whereas
    # SRT is ordered by construction. Restore that invariant so downstream
    # frame-matching sees a monotonic timeline.
    cues.sort(key=lambda c: (c[0], c[1]))
    return cues


def transcribe_openai_api(
    audio: Path, srt_out: Path, model: str | None,
    language: str | None, chunk_seconds: int, workdir: Path,
) -> None:
    resolved = openai_key()
    if not resolved:  # already checked by preflight(); belt and braces
        fail(
            "no OpenAI key found. Set $OPENAI_API_KEY or add a keychain item:\n"
            '  security add-generic-password -s OPENAI_API_KEY -w "<key>"'
        )
        return
    key, _source = resolved
    model = model or MODELS["openai-api"]["fast"]
    diarize = "-diarize" in model
    duration = media_duration(audio)
    if duration <= chunk_seconds:
        chunks = [audio]
    else:
        chunks = segment_audio(audio, workdir, chunk_seconds)
        print(f"chunking: {len(chunks)} chunks of <= {chunk_seconds}s", file=sys.stderr)
    cues: list[tuple[int, int, str]] = []
    offset_ms = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"transcribing chunk {i}/{len(chunks)}", file=sys.stderr)
        raw = _openai_call(chunk, key, model, language)
        parsed = parse_diarized_json(raw) if diarize else parse_srt_ms(raw)
        cues += [(s + offset_ms, e + offset_ms, t) for s, e, t in parsed]
        offset_ms += round(media_duration(chunk) * 1000)
    if not cues:
        fail("API returned no transcript cues (silent audio?)")
    write_srt(cues, srt_out)


def _collect_single_srt(outdir: Path, srt_out: Path) -> None:
    produced = list(outdir.glob("*.srt"))
    if len(produced) != 1:
        fail(f"expected one .srt in {outdir}, found {len(produced)}")
    cues = parse_srt_ms(produced[0].read_text(encoding="utf-8", errors="replace"))
    if not cues:
        fail(f"{produced[0]} contains no cues")
    write_srt(cues, srt_out)


def transcribe_mlx(
    audio: Path, srt_out: Path, model: str | None,
    language: str | None, workdir: Path,
) -> None:
    outdir = workdir / "mlx-out"
    outdir.mkdir(exist_ok=True)
    tool = find_tool("mlx_whisper")
    if not tool:
        fail(
            "mlx_whisper not found on PATH or in the skill's .venv.\n"
            "  pip install mlx-whisper        (Apple Silicon)\n"
            "or install it beside the skill:\n"
            "  python3 -m venv .venv && .venv/bin/pip install mlx-whisper"
        )
    cmd = [
        tool, str(audio),
        "--output-dir", str(outdir), "--output-format", "srt",
        "--model", model or MODELS["mlx-whisper"]["fast"],
    ]
    if language:
        cmd += ["--language", language]
    subprocess.run(cmd, check=True)
    _collect_single_srt(outdir, srt_out)


def transcribe_local_whisper(
    audio: Path, srt_out: Path, model: str | None,
    language: str | None, workdir: Path,
) -> None:
    outdir = workdir / "whisper-out"
    outdir.mkdir(exist_ok=True)
    tool = find_tool("whisper")
    if not tool:
        fail(
            "whisper not found on PATH or in the skill's .venv.\n"
            "  pip install openai-whisper     (any platform)"
        )
    cmd = [
        tool, str(audio),
        "--output_dir", str(outdir), "--output_format", "srt",
        "--model", model or MODELS["openai-whisper"]["fast"],
    ]
    if language:
        cmd += ["--language", language]
    subprocess.run(cmd, check=True)
    _collect_single_srt(outdir, srt_out)


def transcribe_command(
    template: str, video: Path, audio: Path | None, srt_out: Path,
) -> None:
    cmd = template.format(
        video=shlex.quote(str(video)),
        audio=shlex.quote(str(audio)) if audio else "",
        srt=shlex.quote(str(srt_out)),
    )
    print(f"running: {cmd}", file=sys.stderr)
    subprocess.run(cmd, shell=True, check=True)
    if not srt_out.is_file():
        fail(f"command finished but did not produce {srt_out}")


# --- main --------------------------------------------------------------------


def preflight(provider: str, model: str | None) -> None:
    """Reject an unusable provider/model combination before any costly work.

    Runs before audio extraction so a bad `--model` or a missing key does not
    cost a full ffmpeg transcode of the recording first.
    """
    if provider != "openai-api":
        return
    if not openai_key():
        fail(
            "no OpenAI key found. Set $OPENAI_API_KEY or add a keychain item:\n"
            '  security add-generic-password -s OPENAI_API_KEY -w "<key>"'
        )
    if not model:
        return
    reason = untimestamped_reason(model)
    if reason:
        fail(
            f"model `{model}` cannot be used: {reason}.\n"
            "This skill matches every transcript cue to a video frame, so "
            "timestamps are mandatory. Use one of:\n"
            "  gpt-4o-transcribe-diarize  (default; segment timestamps)\n"
            "  whisper-1                  (legacy; native SRT)\n"
            "  a local provider           (mlx-whisper / openai-whisper)"
        )
    if model != "whisper-1" and "-diarize" not in model:
        print(
            f"warning: `{model}` is not a known timestamped model. If the API "
            "rejects response_format=srt, use gpt-4o-transcribe-diarize.",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("media", nargs="?", help="Path to source video/audio")
    ap.add_argument("--detect", action="store_true",
                    help="Report available providers and exit")
    ap.add_argument("--srt-out", default=None,
                    help="Output SRT path (default: alongside the input)")
    ap.add_argument("--provider",
                    choices=["mlx-whisper", "openai-whisper", "openai-api", "command"],
                    default=None, help="Force a provider (default: auto-detect)")
    ap.add_argument("--quality", choices=["fast", "max"], default="fast",
                    help="Quality tier, resolved per provider (default: fast)")
    ap.add_argument("--model", default=None,
                    help="Model override (overrides --quality)")
    ap.add_argument("--language", default=None,
                    help="ISO-639-1 language hint, e.g. en, uk. Leave unset "
                         "for code-switched audio - a forced language degrades "
                         "the other one (default: auto)")
    ap.add_argument("--command", default=None,
                    help="Template for --provider command; "
                         "placeholders {video} {audio} {srt}, pre-quoted")
    ap.add_argument("--chunk-seconds", type=int, default=900,
                    help="API chunk length in seconds (default: 900)")
    args = ap.parse_args()

    if args.detect:
        print_detect_report()
        return

    if not args.media:
        fail("media path required (or use --detect)")
    check_ffmpeg()
    media = Path(args.media).expanduser().resolve()
    if not media.is_file():
        fail(f"input not found: {media}")
    srt_out = (
        Path(args.srt_out).expanduser().resolve()
        if args.srt_out
        else media.with_suffix(".srt")
    )

    provider = args.provider
    if provider is None and args.model:
        # `--model whisper-1` must not be handed to a locally detected
        # mlx_whisper, which would try to fetch it as a HuggingFace repo.
        inferred = provider_for_model(args.model)
        if inferred:
            provider = inferred
            print(
                f"note: --model {args.model} implies --provider {inferred}",
                file=sys.stderr,
            )
    if provider is None:
        available = detect_providers()
        if not available:
            fail(
                "no transcription provider available. Options:\n"
                "  pip install mlx-whisper        (local, Apple Silicon)\n"
                "  pip install openai-whisper     (local, any platform)\n"
                "  set $OPENAI_API_KEY            (OpenAI API, paid)\n"
                "  --provider command --command '<your tool template>'"
            )
        provider = available[0]
    if provider == "command" and not args.command:
        fail("--provider command requires --command TEMPLATE")

    model = args.model or (
        MODELS[provider][args.quality] if provider in MODELS else None
    )
    # Validate before extract_audio() so a rejected model or a missing key
    # costs nothing - a full ffmpeg pass over a long recording is minutes.
    preflight(provider, model)
    print(f"provider: {provider}" + (f" / {model}" if model else ""), file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="video-to-spec-") as td:
        workdir = Path(td)
        if provider == "command":
            audio = (
                extract_audio(media, workdir) if "{audio}" in args.command else None
            )
            transcribe_command(args.command, media, audio, srt_out)
        else:
            audio = extract_audio(media, workdir)
            if provider == "openai-api":
                transcribe_openai_api(
                    audio, srt_out, model, args.language,
                    args.chunk_seconds, workdir,
                )
            elif provider == "mlx-whisper":
                transcribe_mlx(audio, srt_out, model, args.language, workdir)
            else:
                transcribe_local_whisper(
                    audio, srt_out, model, args.language, workdir
                )

    print(f"wrote: {srt_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
