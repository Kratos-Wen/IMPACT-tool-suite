# IMPACT tool suite

A single desktop application for editing hand–object interaction annotations and testing optional SAM2 correction propagation. The English interface is retained.

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
