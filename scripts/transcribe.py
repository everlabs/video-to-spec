#!/usr/bin/env python3
"""Generate an SRT transcript for a video (or audio) file.

Providers, auto-detected in preference order (local tools first):
    mlx-whisper     `mlx_whisper` CLI - local, free, fast on Apple Silicon
    openai-whisper  `whisper` CLI - local, free
    openai-api      OpenAI transcription API (whisper-1, native SRT output).
                    Key from $OPENAI_API_KEY, else the macOS keychain
                    (generic password with service name OPENAI_API_KEY).
    command         any user-supplied template, e.g.
                    --provider command --command 'mytool {audio} -o {srt}'
                    Placeholders {video} {audio} {srt} are substituted
                    shell-quoted. If {audio} appears, a 16 kHz mono mp3 is
                    extracted first.

Usage:
    transcribe.py --detect
    transcribe.py <video> [--srt-out PATH] [--provider P] [--model M]
                  [--language xx] [--command TEMPLATE] [--chunk-seconds N]

The API path chunks long audio (default 900 s per chunk, ~3.5 MB at 32 kbps,
safely under the 25 MB upload limit), transcribes each chunk, offsets the
timestamps by the real duration of the preceding chunks, and merges into one
renumbered SRT.
"""

from __future__ import annotations

import argparse
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


def detect_providers() -> list[str]:
    available = []
    if shutil.which("mlx_whisper"):
        available.append("mlx-whisper")
    if shutil.which("whisper"):
        available.append("openai-whisper")
    if openai_key():
        available.append("openai-api")
    return available


def print_detect_report() -> None:
    key = openai_key()
    rows = [
        ("mlx-whisper", "available" if shutil.which("mlx_whisper") else "not found"),
        ("openai-whisper", "available" if shutil.which("whisper") else "not found"),
        ("openai-api", f"available (key in {key[1]})" if key else "not found"),
    ]
    for name, status in rows:
        print(f"{name}: {status}")
    for extra in ("whisper-cli", "whisper-cpp"):
        if shutil.which(extra):
            print(f"note: `{extra}` is on PATH - usable via --provider command")
    usable = detect_providers()
    print(f"usable: {', '.join(usable) if usable else 'none'}")


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
    fields = {"model": model, "response_format": "srt"}
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
    return ""  # unreachable


def transcribe_openai_api(
    audio: Path, srt_out: Path, model: str | None,
    language: str | None, chunk_seconds: int, workdir: Path,
) -> None:
    resolved = openai_key()
    if not resolved:
        fail(
            "no OpenAI key found. Set $OPENAI_API_KEY or add a keychain item:\n"
            '  security add-generic-password -s OPENAI_API_KEY -w "<key>"'
        )
    key, _source = resolved
    model = model or "whisper-1"
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
        srt_text = _openai_call(chunk, key, model, language)
        cues += [(s + offset_ms, e + offset_ms, t) for s, e, t in parse_srt_ms(srt_text)]
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
    cmd = [
        "mlx_whisper", str(audio),
        "--output-dir", str(outdir), "--output-format", "srt",
        "--model", model or "mlx-community/whisper-large-v3-turbo",
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
    cmd = [
        "whisper", str(audio),
        "--output_dir", str(outdir), "--output_format", "srt",
        "--model", model or "turbo",
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
    ap.add_argument("--model", default=None,
                    help="Model override (per-provider default otherwise)")
    ap.add_argument("--language", default=None,
                    help="ISO-639-1 language hint, e.g. en, uk (default: auto)")
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

    print(f"provider: {provider}", file=sys.stderr)
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
                    audio, srt_out, args.model, args.language,
                    args.chunk_seconds, workdir,
                )
            elif provider == "mlx-whisper":
                transcribe_mlx(audio, srt_out, args.model, args.language, workdir)
            else:
                transcribe_local_whisper(
                    audio, srt_out, args.model, args.language, workdir
                )

    print(f"wrote: {srt_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
