# 安装与启动

仓库只有一个软件版本。下载后直接在仓库根目录安装和启动，不选择工程批号目录。

## 1. 获取代码

安装 Git 和 Python 3.10/3.11。以下命令示例使用 Python 3.11。终端运行：

```bash
git clone https://github.com/Kratos-Wen/IMPACT-tool-suite.git
cd IMPACT-tool-suite
```

也可以在 GitHub 选择 Code → Download ZIP，解压后进入包含 `app.py` 的目录。

## 2. 安装普通编辑界面

Windows PowerShell（使用环境中的 Python，不依赖激活脚本权限）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch.py --oplog
```

Linux / macOS：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python launch.py --oplog
```

普通拖框、事件编辑和保存不需要安装 SAM。`requirements-optional.txt` 是历史辅助功能的可选依赖，不是 SAM 安装步骤，也不必为普通编辑全部安装。Linux 若报 `libGL.so.1` 等缺失，按发行版安装系统 Qt/OpenGL 库（Ubuntu 可安装 `libgl1 libegl1 libxkbcommon-x11-0 libxcb-xinerama0`）。远程纯终端没有桌面显示时不能直接交互使用 GUI。

## 3. 安装 SAM2.1 Small 修正传播

建议在 Linux，或 Windows 11 的 WSL2 Ubuntu 桌面显示环境中同时运行工具和 SAM。Windows 原生及 macOS 的 SAM 后端未完成验收；不能把本节当作这些平台已验证的安装承诺。WSL 中创建并使用 Linux 的虚拟环境，不能复用 Windows 的 `.venv`。

进入仓库根目录，激活 Linux 环境：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

CPU 安装（功能可运行，传播会慢）：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

有兼容 NVIDIA GPU、驱动且希望用 GPU 时，改用下面这条，不要同时执行 CPU 安装：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

GPU 模式应输出 `True`。否则先检查驱动和 PyTorch 安装，不要直接启用 GPU。工具不会申请 Slurm GPU。

安装已测试的 SAM2 源码：

```bash
mkdir -p .deps
git clone https://github.com/facebookresearch/sam2.git .deps/sam2
git -C .deps/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e .deps/sam2
python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM import OK')"
```

`SAM2_BUILD_CUDA=0` 省去编译可选 CUDA 后处理扩展，不关闭 GPU 推理；相关小孔/碎片后处理会跳过。具体要求见 [SAM2 官方安装说明](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)。

取得 `sam2.1_hiera_small.pt`：可以使用项目资产包已附权重，或从 [官方地址](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) 下载。权重留在 Git 仓库外，或放在被忽略的 `weights/` 目录。运行时无需联网下载模型。

启动 SAM 界面（将路径改成实际本地路径）：

```bash
python launch.py --sam-checkpoint "/absolute/path/sam2.1_hiera_small.pt" --device cpu --oplog
```

已验证 CUDA 可用时，将 `--device cpu` 改成 `--device cuda`。若后端放在独立环境，还可添加 `--sam-python "/absolute/path/to/backend/python"`。工具和后端均在同一 Windows 或同一 WSL/Linux 文件系统环境运行。

## 4. 使用项目资产

项目负责人另外发放视频、预标注、权重、项目配置和标注规范。这些文件不在公开仓库内。项目配置不会自动下载。

```bash
python launch.py --project-profile "/absolute/path/project_profile.json" --sam-checkpoint "/absolute/path/sam2.1_hiera_small.pt" --device cpu --oplog
```

Windows 普通编辑时同样支持 `--project-profile`：

```powershell
.\.venv\Scripts\python.exe launch.py --project-profile "C:\IMPACT_assets\project_profile.json" --oplog
```

先选 Manual 模式打开视频，再使用 Load HOI annotations 导入与视频配套的 JSON。如果项目包包含 `SAM_editor_candidates.json`，该文件用于主界面的传播功能测试；`review.json` 打开独立预标注审核界面，该界面尚未接入 Track Correction。

## 5. 软件操作

选中事件及其物品/工具，在当前帧拖框或改变框大小，然后点击 **Track Correction**，输入要处理的后续帧数。检查建议后接受，或取消。接受后可撤销。传播受事件边界及既有人工锚点保护。

Ctrl+滚轮缩放；右键框只删除当前帧的框；修改时间轴标记可调整事件边界。完成后另存 JSON，重新打开检查保存结果。具体哪些帧需要人工标注、如何定义事件和类别，由项目内部规范决定，公开文档不规定这些要求。

## 6. 故障信息与验证范围

- 缺依赖：确认在同一个 `.venv` 中安装和启动。
- SAM import OK 失败：检查固定版本源码是否安装到启动所用 Python 中。
- 找不到权重/配置：启动参数必须对应已下载文件，建议用绝对路径。
- CUDA 不可用：检查上一节命令，或明确使用 CPU。
- Qt 插件错误：避免混用 Conda、系统 Qt 和 venv；使用本工具启动入口。
- 当前记录包含 Linux Qt 的组件测试和短视频传播测试；完整原主窗口流程及 Windows/macOS 尚未全部验收，部署测试结论见 `VALIDATION.md`。
