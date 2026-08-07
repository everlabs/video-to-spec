# video-to-spec

An [Agent Skill](https://agentskills.io) that turns a screen-recording walkthrough (Loom, QuickTime, anything ffmpeg reads) into a folder of reviewable, self-contained user-story specifications – each task with the exact screenshots the speaker was looking at when they said it.

Works with any AI agent that supports the SKILL.md standard: Claude Code, Codex, Cursor, OpenCode, Gemini CLI, and others.

```
recording.mp4 (+ optional captions)
        │
        ▼
frames @ 1fps ──► perceptual-hash dedup ──► unique screen states
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

Just feed this skill to your AI agent: copy the `video-to-spec` folder into your agent's skills directory.

```bash
git clone <this-repo>
cp -R public-skills/video-to-spec ~/.claude/skills/    # Claude Code
cp -R public-skills/video-to-spec ~/.codex/skills/     # Codex
```

(Cursor, OpenCode, Gemini CLI, etc. – same folder, your agent's skills path. Or simply tell your agent to install it from this repo.)

Then say things like *"video to spec"*, *"process my loom feedback"*, *"turn this walkthrough into tasks"*, and point your agent at a video.

## Requirements

| Dependency | Needed for | Install |
|---|---|---|
| ffmpeg | always (frames, audio) | `brew install ffmpeg` |
| ImageMagick *or* Python `imagehash` | frame dedup (auto-detected) | `brew install imagemagick` |
| a transcription provider | only when the video has no captions | see below |

If the video comes with an `.srt`/`.vtt` (Loom exports these), no transcription setup is needed at all.

## Transcription providers

Auto-detected, local tools preferred (free, private). The skill reports what it found and confirms before any paid API call.

| Provider | Needs | Notes |
|---|---|---|
| `mlx-whisper` | `pip install mlx-whisper` | Apple Silicon, fast |
| `openai-whisper` | `pip install openai-whisper` | any platform |
| `openai-api` | `$OPENAI_API_KEY` (or macOS keychain item `OPENAI_API_KEY`) | ~$0.006/min, long files auto-chunked |
| `command` | your own tool | `--command 'mytool {audio} -o {srt}'` |

## What review looks like

The skill drafts everything first, then asks its clarifying questions as one batch – ambiguities, contradictions, places where it read an implied requirement beyond what was literally said. You answer once, the affected specs get updated, and the folder is ready to hand off.

## Credits

Built by **Oleg Pasko** @ **Everlabs**. Feedback and PRs welcome.
