# IMPACT tool suite

A single desktop application for editing hand–object interaction annotations and testing optional SAM2 correction propagation. The English interface is retained.

## Installation

See [详细中文安装、启动及操作说明](docs/INSTALL_zh.md). Run from the repository root; no engineering-batch or variant directory is required.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python launch.py --oplog
```

Windows and the optional CPU/CUDA SAM2 backend have separate commands in the installation guide. SAM inference needs the matching checkpoint and installed backend, not just GUI dependencies.

## Project assets

Videos, model weights, preannotations, project profiles, taxonomies and annotation requirements are distributed separately by the project coordinator. They are not included in this public repository. Configure a local profile with `--project-profile`; the software does not retrieve it automatically.

## Development checks

```bash
python tests/test_instance_edits.py
python tests/test_correction_propagation.py
python tests/test_noun_aliases.py
python tests/test_orientation_guard.py
```

See [validation and current limitations](VALIDATION.md). This is a functional test release; automated proposals are not verified ground truth. Upstream model licenses apply to their respective models.
