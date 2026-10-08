# Install and run

Use Python 3.11 and Git. Linux, Windows and macOS use the same application.

```bash
git clone https://github.com/Kratos-Wen/IMPACT-tool-suite.git
cd IMPACT-tool-suite
```

## Desktop editor

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch.py --oplog
```

Linux and macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python launch.py --oplog
```

On Ubuntu, install Qt libraries if missing:

```bash
sudo apt-get install libgl1 libegl1 libxkbcommon-x11-0 libxcb-xinerama0
```

## SAM correction propagation on CPU

Use the same virtual environment and install build tools with `python -m pip install wheel setuptools`. Install PyTorch on Linux/Windows:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1
```

In these commands, `python` means `.venv/bin/python` on Linux/macOS or `.\.venv\Scripts\python.exe` on Windows. Install SAM:

```bash
git clone https://github.com/facebookresearch/sam2.git .deps/sam2
git -C .deps/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Linux/macOS:

```bash
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --no-build-isolation -e .deps/sam2
```

Windows PowerShell:

```powershell
$env:SAM2_BUILD_CUDA = "0"
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .deps/sam2
```

Use the supplied `sam2.1_hiera_small.pt`, or download the [official checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt). For standalone work, pass `--sam-checkpoint "/path/to/sam2.1_hiera_small.pt" --device cpu` to `launch.py`. A packaged task supplies this path automatically. A separately installed backend can be selected with `--sam-python`.

## Load, edit and save

Download your task directory and its sibling `Shared`. Choose Manual mode and open a task’s `video.mp4`. The neighboring `task.json` loads the local project profile and editor draft. Existing `reviewed.json` resumes your work; Save defaults to that file. The manifest checks trial identity, FPS and frame count.

For a standalone video, use `--project-profile "/path/to/project_profile.json"` and Load HOI annotations to select its draft. Legacy editor imports preserve the source in `archived` before conversion.

Select an event and instance, drag or resize the box, then use **Track Correction** to propagate to following or preceding frames. Check proposals before applying them. Ctrl+wheel zooms; right-click deletes the current box. Frame review controls verify boxes and navigate unfinished work. Undo restores edits and propagation. Project annotation requirements are supplied in the private task materials.

## Troubleshooting

Install and launch with the same environment’s Python. A missing SAM module requires the backend installation above. Use an existing local checkpoint path. For Qt plugin errors, use `launch.py` and avoid mixing system Qt, Conda and virtual-environment libraries. See [automated validation](../VALIDATION.md) for test coverage.
