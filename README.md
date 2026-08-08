# video-to-spec

An [Agent Skill](https://agentskills.io) that turns a screen-recording walkthrough (Loom, QuickTime, anything ffmpeg reads) into a folder of reviewable, self-contained user-story specifications – each task with the exact screenshots the speaker was looking at when they said it.

Works with any AI agent that supports the SKILL.md standard: Claude Code, Codex, Cursor, OpenCode, Gemini CLI, and others.

<table>
  <tr>
    <th>How it works</th>
    <th>What you get</th>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="assets/video-to-spec-scheme.svg" alt="How the skill works: a screen recording is split into deduplicated frames and a transcript, merged into a timeline, mined for tasks, and written out as a numbered spec folder.">
    </td>
    <td width="50%" valign="top">
      <img src="assets/video-to-spec-example.svg" alt="A single spec file produced by the skill: task title, type and source timestamps, user story, the quoted remark it came from, and a wireframe of the screen it refers to.">
    </td>
  </tr>
</table>

```
recording.mp4 (+ optional captions)
        │
        ▼
frames @ 1fps ──► pixel-diff dedup ──► unique screen states
transcript    ──► (provided SRT/VTT, or generated via whisper)
        │
        ▼
timeline: every spoken segment matched to the screen it was said on
        │
        ▼
docs/video-to-spec/<video>--<date>/
├── index.md            summary + open questions
├── 01-<task>.md        user story · motivation · screenshots · what to do
├── 02-<task>.md
└── frames/
```

## Why this exists

A lot of my product feedback happens through Loom. The workflow that stuck: hit record, open the service I want to improve, and think out loud while browsing it – ideas, brainstorming, small annoyances, half-formed feature requests. That recording is the most honest snapshot of my product vision, but it isn't actionable: it's twenty minutes of "hmm, this is weird" and "what if we…" scattered across screens.

This skill closes that gap. It decomposes the walkthrough into clarified specifications, confirms what I actually said (or surfaces where I said something different from what I remember meaning), asks a single batch of clarifying questions, and writes specs clean enough to delegate – to sub-agents, to AI coding agents, or to a human team – so the code gets written against the product vision, not against someone's reconstruction of it.

The extraction is opinionated about how people actually talk in walkthroughs:

- **Direct tasks** – "let's fix…", "this should…"
- **Reversals** – "wait, actually, scratch that" – a later contradiction always wins
- **Dissatisfaction** – "this is weird" becomes a task only when the change can be named, otherwise an open question
- **Indirect requirements** – problems visible in the frame the speaker is describing but never explicitly requests; the screenshot is treated as the source of truth

## Install

This repository *is* the skill, so clone it straight into your agent's skills directory:

```bash
git clone https://github.com/everlabs/video-to-spec.git ~/.claude/skills/video-to-spec   # Claude Code
git clone https://github.com/everlabs/video-to-spec.git ~/.codex/skills/video-to-spec    # Codex
```

Cursor, OpenCode, Gemini CLI, etc. – same command, your agent's skills path. Or simply tell your agent to install it from this repo.

Update later with `git -C ~/.claude/skills/video-to-spec pull`. The virtualenv described below lives in that folder and is git-ignored, so pulling never disturbs it.

Then say things like *"video to spec"*, *"process my loom feedback"*, *"turn this walkthrough into tasks"*, and point your agent at a video.

## Requirements

| Dependency | Needed for | Install |
|---|---|---|
| ffmpeg | always (frames, audio, dedup) | `brew install ffmpeg` |
| a transcription provider | only when the video has no captions | see below |

Frame dedup needs nothing beyond ffmpeg: the default `pixel` backend compares 128×128 grayscale thumbnails and keeps a frame when more than 0.1% of pixels change from the last kept one. ImageMagick PHASH and Python `imagehash` remain selectable with `--backend`, but they are a poor fit for screen recordings – perceptual hashing is built to *ignore* small visual changes, and a modal opening or a form field filling in is exactly that. They are also far slower, spawning a process per frame pair.

If the video comes with an `.srt`/`.vtt` (Loom exports these), no transcription setup is needed at all.

## Transcription providers

Auto-detected, local tools preferred (free, private, and on Apple Silicon also the fastest). The skill reports what it found and confirms before any paid API call.

| Provider | Platform | Needs | Default model | `--quality max` |
|---|---|---|---|---|
| `mlx-whisper` | Apple Silicon only | `pip install mlx-whisper` | `mlx-community/whisper-large-v3-turbo` | `mlx-community/whisper-large-v3-mlx` |
| `openai-whisper` | any | `pip install openai-whisper` | `turbo` | `large-v3` |
| `openai-api` | any | `$OPENAI_API_KEY` (or macOS keychain item `OPENAI_API_KEY`) | `gpt-4o-transcribe-diarize` | – |
| `command` | any | your own tool: `--command 'mytool {audio} -o {srt}'` | – | – |

`mlx-whisper` has no wheel for Intel Macs or Linux – use `openai-whisper` there.

Installing into a virtualenv beside the skill keeps a multi-gigabyte ML dependency out of your system Python. The script looks in `<skill>/.venv` (and `<skill>/venv`) as well as on `PATH`, so no activation is needed – but it must be *inside the folder this skill was installed to*, whichever agent that is:

```bash
cd /path/to/your/skills/video-to-spec     # wherever your agent installed it
python3 -m venv .venv && .venv/bin/pip install mlx-whisper
python3 scripts/transcribe.py --detect    # confirms it was found
```

### Choosing a model

**Local is the recommendation, not the fallback.** On Apple Silicon `large-v3-turbo` transcribes about 10× faster than real time, costs nothing, and never uploads the recording.

`--quality max` (`large-v3`) is a **retry, not an upgrade**. It runs roughly 20× slower for a result that is different rather than reliably better: in side-by-side runs on mixed English/Ukrainian audio, each model won some segments and lost others – `large-v3` handled mid-sentence language switches more often, `turbo` was cleaner on single-language passages. Transcribe fast, skim, and re-run at `max` only if the transcript looks garbled.

**Don't set a language on code-switched recordings.** Forcing one language degrades the other; auto-detection handles mixed audio better than a wrong hint.

### Why not `gpt-transcribe`?

It is OpenAI's most accurate transcription model, and it cannot be used here. This skill pins every transcript cue to a video frame, so segment timestamps are mandatory, and `gpt-transcribe` returns JSON/text with none – the API rejects `response_format=srt` outright. The script fails fast with an explanation rather than silently degrading.

`gpt-4o-transcribe-diarize` is the only current-generation OpenAI model that emits per-segment `start`/`end`, so it is the cloud default: the script requests `diarized_json` and converts it to SRT. It is token-priced, so the per-minute rate depends on the audio – OpenAI's published estimate is ~$0.006/min, while a measured run on mixed English/Ukrainian came out at ~$0.02/min, since Cyrillic costs roughly 3× the output tokens of Latin script. Legacy `whisper-1` remains available via `--provider openai-api --model whisper-1` at a flat $0.006/min, but it renders Ukrainian in visibly Russified spelling – a compatibility fallback, not a recommendation.

## What review looks like

The skill drafts everything first, then asks its clarifying questions as one batch – ambiguities, contradictions, places where it read an implied requirement beyond what was literally said. You answer once, the affected specs get updated, and the folder is ready to hand off.

## Contributing

Issues and pull requests are welcome – this repo holds nothing but the skill, so a change here cannot break anything else.

The moving parts, in the order they run:

| File | Role |
|---|---|
| `SKILL.md` | the whole procedure your agent follows – 10 numbered steps, plus the rules for turning speech into tasks |
| `scripts/process_inputs.py` | frames, dedup, transcript parsing, and the merged `timeline.md` |
| `scripts/transcribe.py` | provider detection and transcription, only when no captions were supplied |

Both scripts are dependency-free Python 3 (standard library plus ffmpeg on `PATH`), so they run with any `python3` and need no build step. `SKILL.md` is prose, and changes there are best tested by pointing an agent at a real recording and watching where it goes wrong.

Two things to keep in mind:

- **The skill must stay agent-agnostic.** No harness-specific tool names in `SKILL.md` – say "your todo tool", not the name of one particular agent's.
- **Dedup thresholds are empirical.** If you change `--backend` defaults or the pixel constants, report what you measured them on; screen recordings vary enormously between desktop and portrait mobile capture.

## Credits

Built by **Oleg Pasko** @ **Everlabs**. One of the [Everlabs public skills](https://github.com/everlabs/public-skills).
