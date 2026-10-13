# IMPACT tool suite

[![Desktop](https://github.com/Kratos-Wen/IMPACT-tool-suite/actions/workflows/desktop.yml/badge.svg?branch=main)](https://github.com/Kratos-Wen/IMPACT-tool-suite/actions/workflows/desktop.yml)
[![Audio](https://github.com/Kratos-Wen/IMPACT-tool-suite/actions/workflows/audio-review.yml/badge.svg?branch=main)](https://github.com/Kratos-Wen/IMPACT-tool-suite/actions/workflows/audio-review.yml)
[![SAM CPU](https://github.com/Kratos-Wen/IMPACT-tool-suite/actions/workflows/sam-cpu.yml/badge.svg?branch=main)](https://github.com/Kratos-Wen/IMPACT-tool-suite/actions/workflows/sam-cpu.yml)

A single desktop application for editing hand–object interaction annotations and testing optional SAM2 correction propagation. The English interface is retained.

The separate [offline audio review utility](audio_review/README.md) opens in a browser. It does not require the desktop application's dependencies.

## Installation

See the [installation guide](docs/INSTALL_en.md) or [中文说明](docs/INSTALL_zh.md). Run from the repository root.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python launch.py --oplog
```

Windows and the optional CPU/CUDA SAM2 backend have separate commands in the installation guide. SAM inference needs the matching checkpoint and installed backend, not just GUI dependencies.

## Project assets

Videos, model weights, preannotations, project profiles, taxonomies and annotation requirements are distributed separately by the project coordinator. They are not included in this public repository. Opening a packaged video loads its local profile and draft through `task.json`. For standalone work, configure a profile with `--project-profile`.

## Development checks

```bash
python tests/test_instance_edits.py
python tests/test_correction_propagation.py
python tests/test_noun_aliases.py
python tests/test_orientation_guard.py
```

See [validation and current limitations](VALIDATION.md). Tracking proposals require human review. Upstream model licenses apply to their respective models.

### Saving and recovery

Task edits are saved to a hidden recovery folder every two seconds, including unfinished events, boxes, accepted tracking, shared state and review records. Reopen the same video after an unexpected exit and accept the recovery prompt. The autosave status shows the last successful write. Save the reviewed handoff as `reviewed.json` beside the task video. Writes replace files atomically, and a second editor cannot write the same task concurrently.

The **Noun and Verb Guide** in the file menu reads explanations from your project profile. Selector tooltips show the same definitions; the guide supports search and English/Chinese text supplied by the project. Added labels remain visible and require an agreed project definition.

Project profiles can restrict noun choices by the video's model. Incompatible options are disabled. Imported conflicts retain their IDs and geometry and appear in the completion checks for correction. Confirmed class aliases apply on import and save.

**Assembly composition** in the file menu edits one object ID at a time. Multiple instances can have different compositions at the same frame. Hands referencing the same ID share that instance's state. **Merge assemblies** records a completed connection: choose the inserted instance, the receiving ID and the connection frame, then correct the receiving instance's whole-object box. Its correction propagation stays inside the current composition interval. The inserted track ends at the merge; event geometry links follow the receiving track. Composition changes and merges support undo, recovery and save/load.

The next launch reopens the previous local project and playback position. Task manifests load matching assets and prefer `reviewed.json`; newer recovery drafts still offer a restore prompt. Missing files produce a relocation message. Set `IMPACT_RESTORE_LAST_PROJECT=0` to start with an empty editor.

**Track Correction** opens a target checklist and frame interval. Choose the current event's hands, Object and instrument, then set the beginning and end on either side of the current anchor frame. Targets need an unambiguous box at that frame. Each branch respects its event, human anchors and assembly changes.

Propagation runs in a background queue. Continue editing while the small progress bar shows processed target-frames and an estimated remaining time. Results automatically become **unverified** boxes; each applied batch supports one-step undo. Manual boxes, deletions, visibility decisions and changed event bindings are preserved. A new manual reference inside a running interval cuts the old result and schedules a bounded continuation from that reference. **Details** lists jobs and offers cancel, retry/resume and recovery of saved candidates.

Tasks and partial candidates are saved beside the video in `.impact-recovery`. Reopening the same video restores interrupted tasks. Completed branches are reused; an unfinished branch restarts from its reference frame. The resident backend reuses the SAM model and groups targets sharing a reference frame and direction into one video state. It releases the model after two idle minutes. CPU propagation uses up to four threads by default, leaving two logical CPUs available when possible; override with `IMPACT_SAM2_CPU_THREADS`.

Thin yellow lines inside the selected event mark frames with boxes for its hand, Object or instrument. Orange lines indicate duplicate or invalid boxes. Click a line to seek to that frame; hover shows the entity and box source. Dragging still adjusts the event; use the top of the block to drag Start/Onset/End handles. Onset retains its upward arrow. Dense tracks share a line when several frames occupy the same screen pixel; zoom in or use **Alt+, / Alt+.** to visit each box frame.

Over the timeline, **Ctrl/Command + wheel** zooms at the pointer and **Shift + wheel** pans. Over the video, existing image zoom is unchanged. **Alt+1/2/3** jumps to the selected event's Start/Onset/End; **Alt+Left/Right** jumps through its event and review keyframes. **Alt+N/P** continues to find frames needing review. **Ctrl+Enter** verifies the visible boxes on the current frame. The **File → Keyframes** menu lists these shortcuts.

Event intervals include both Start and End frames. Events for the same hand must not overlap or share a boundary frame: after End 65, the next Start is at least 66. Start, Onset and End within one event may coincide. Left-hand and right-hand events may overlap. Existing conflicting annotations are retained and flagged for correction; timestamps are not shifted automatically.

Bbox frames appear as thin yellow lines inside the selected event; orange lines indicate duplicate or invalid boxes. Click a line to seek to that frame. These lines never move event boundaries, even if the pointer moves during the click. Drag the upper SOE handles to adjust timing. Importing sparse or hands-only annotations retains the active project vocabulary.
