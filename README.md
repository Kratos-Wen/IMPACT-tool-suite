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

The current project configuration and taxonomy are versioned in [project_profile.json](project_profile.json). Startup, task imports and remembered sessions use this repository copy, including when an older task points to Shared. Run `git pull` and restart the editor to update both code and configuration; no separate Shared configuration download is needed. Videos, model weights, preannotations and detailed annotation requirements remain separately distributed. Task manifests still locate each video’s annotations and checkpoints.

## Development checks

Synthetic fixture tests use `IMPACT_USE_BUNDLED_PROFILE=0`, as in CI. `test_repository_profile.py` enables and tests the shipped configuration. External custom projects can also explicitly set this flag to use `--project-profile`; normal annotation uses the repository copy.

```bash
IMPACT_USE_BUNDLED_PROFILE=0 python tests/test_instance_edits.py
IMPACT_USE_BUNDLED_PROFILE=0 python tests/test_correction_propagation.py
IMPACT_USE_BUNDLED_PROFILE=0 python tests/test_noun_aliases.py
IMPACT_USE_BUNDLED_PROFILE=0 python tests/test_orientation_guard.py
```

See [validation and current limitations](VALIDATION.md). Tracking proposals require human review. Upstream model licenses apply to their respective models.

### Saving and recovery

Task edits are saved to a hidden recovery folder every two seconds, including unfinished events, boxes, accepted tracking, shared state and review records. Reopen the same video after an unexpected exit and accept the recovery prompt. The autosave status shows the last successful write. Save the reviewed handoff as `reviewed.json` beside the task video. Writes replace files atomically, and a second editor cannot write the same task concurrently.

The **Noun and Verb Guide** in the file menu reads explanations from your project profile. Selector tooltips show the same definitions; the guide supports search and English/Chinese text supplied by the project. Added labels remain visible and require an agreed project definition.

Project profiles can restrict noun choices by the video's model. Incompatible options are disabled. Imported conflicts retain their IDs and geometry and appear in the completion checks for correction. Confirmed class aliases apply on import and save.

Set `instrument_classes` in a project profile to restrict Instrument selectors to tools. Tools remain available as Object for actions such as pickup. An invalid imported Instrument remains marked for correction; refreshing a selector does not erase it. Set `compact_import_ids: true` to number sparse Object IDs consecutively on first import. Saved local IDs remain stable, and hand detector IDs do not consume Object numbers.

To import just the hands from an annotation JSON, open the video, then use **File → Import → Hands from preannotation JSON…**. This preserves existing events, Object IDs, manual boxes and deleted-hand records. Conflicting hand candidates are omitted rather than assigned a guessed identity. **HOI Annotations…** imports the complete annotation project.

**Assembly composition** in the file menu edits one object ID at a time. Multiple instances can have different compositions at the same frame. Hands referencing the same ID share that instance's state. **Merge assemblies** records a completed connection: choose the inserted instance, the receiving ID and the connection frame, then correct the receiving instance's whole-object box. Its correction propagation stays inside the current composition interval. The inserted track ends at the merge; event geometry links follow the receiving track. Composition changes and merges support undo, recovery and save/load.

The next launch reopens the previous local project and playback position. Task manifests load matching assets and prefer `reviewed.json`; newer recovery drafts still offer a restore prompt. Missing files produce a relocation message. Set `IMPACT_RESTORE_LAST_PROJECT=0` to start with an empty editor.

**Track Correction** opens a target checklist and frame interval. Choose the current event's hands, Object and instrument, then set the beginning and end on either side of the current anchor frame. Targets need an unambiguous box at that frame. Each branch respects its event, human anchors and assembly changes.

Propagation runs in a background queue. Continue editing while the small progress bar shows processed target-frames and an estimated remaining time. Results automatically become **unverified** boxes; each applied batch supports one-step undo. Manual boxes, deletions, visibility decisions and changed event bindings are preserved. A new manual reference inside a running interval cuts the old result and schedules a bounded continuation from that reference. **Details** lists jobs and offers cancel, retry/resume and recovery of saved candidates.

Tasks and partial candidates are saved beside the video in `.impact-recovery`. Reopening the same video restores interrupted tasks. Completed branches are reused; an unfinished branch restarts from its reference frame. The resident backend reuses the SAM model and groups targets sharing a reference frame and direction into one video state. It releases the model after two idle minutes. CPU propagation uses up to four threads by default, leaving two logical CPUs available when possible; override with `IMPACT_SAM2_CPU_THREADS`.

Thin yellow lines inside the selected event mark frames where a person drew or edited a hand, Object or instrument box. Imported machine tracking and merely viewed proposals do not create lines. Orange lines indicate duplicate or invalid manual boxes. Click a line to seek; hover shows the entity and box source. These lines never drag an event. Use the top of the block to drag Start/Onset/End handles; Onset retains its upward arrow. **Alt+, / Alt+.** visits the previous/next manually annotated box frame.

Machine hand candidates appear only at the selected event's Start/Onset/End, explicitly added keyframes and manually annotated bbox frames. Manual hand boxes remain visible on their own frames. Dense candidates remain available in saved files for later tracking.

Over the timeline, **Ctrl/Command + wheel** zooms at the pointer and **Shift + wheel** pans. Over the video, existing image zoom is unchanged. **Alt+1/2/3** jumps to the selected event's Start/Onset/End; **Alt+Left/Right** jumps through its event and review keyframes. **Alt+N/P** continues to find frames needing review. **Ctrl+Enter** verifies the visible boxes on the current frame. The **File → Keyframes** menu lists these shortcuts.

Event intervals include both Start and End frames. Events for the same hand must not overlap or share a boundary frame: after End 65, the next Start is at least 66. Start, Onset and End within one event may coincide. Left-hand and right-hand events may overlap. Existing conflicting annotations are retained and flagged for correction; timestamps are not shifted automatically.

Importing sparse or hands-only annotations retains the active project vocabulary.

A newly drawn manual entity stays visible and editable on its creation frame before it is assigned to the event. Select its ID under **Noun** or **Instrument** to bind it. Label entry and instance selectors suggest substring and spelling matches. Click a suggestion; in the box-label dialog, **Tab** accepts the highlighted suggestion. Exact names and explicit instance IDs take precedence over suggestions.

The editor uses one manual annotation workflow. Project videos load their configuration and existing annotations; imported events and boxes remain editable. SAM correction from a reference box is an optional explicit action. Local automatic detection, action ranking and mode switching are no longer part of the desktop workflow. Saved projects using older modes reopen in the manual workflow.

Vocabulary loads automatically from the repository configuration, including for standalone videos. **Import → Reload project configuration** rereads that file without replacing existing annotations. After creating an event, its noun may be selected before its verb. Hand-box record IDs use a separate namespace and do not set the next object/assembly ID.

Each video has two anatomical hand entities, `H_LEFT` and `H_RIGHT`. Imports, manual boxes and correction propagation retain the same hand identity across frames and events. Hand IDs are separate from object instance IDs. XML exports contain one track per side with stable IDs 0 (left) and 1 (right); JSON exports record the named hand IDs. Existing recovery geometry is normalized on reload.

A task may also specify `"hand_tracks": "hands.xml"`. Opening its video loads this optional file automatically after the annotation or recovery data. The file must carry matching video identity, frame count and FPS; no time offset is inferred. Existing boxes, deleted-hand records and invisible-hand decisions take precedence. These boxes remain machine candidates and use the same event/keyframe visibility rules. Download the updated `task.json` together with `hands.xml`.

Counted instance states can be configured in the repository project profile using `counted_object_states`. Selecting a configured Object opens its state editor; a nearby button reopens it. Counts follow the same physical ID over time and are included in save, recovery and undo. Creating a separated piece explicitly assigns a new ID with a parent reference; changing its state keeps the ID. New pieces appear in Noun without changing existing event references; select the new ID for the appropriate later event.
