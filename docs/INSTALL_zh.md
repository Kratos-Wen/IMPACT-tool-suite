# 安装与使用

使用Python 3.11和Git。Windows、Linux、macOS运行同一套工具。[完整安装命令](INSTALL_en.md)。

## 安装编辑器

Windows PowerShell：

```powershell
git clone https://github.com/Kratos-Wen/IMPACT-tool-suite.git
cd IMPACT-tool-suite
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch.py --oplog
```

Linux/macOS：

```bash
git clone https://github.com/Kratos-Wen/IMPACT-tool-suite.git
cd IMPACT-tool-suite
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python launch.py --oplog
```

## 安装SAM修正传播

按照[英文安装说明](INSTALL_en.md#sam-correction-propagation-on-cpu)在同一虚拟环境安装PyTorch和固定版本的SAM2。Windows/Linux使用CPU wheel索引，macOS使用普通PyPI wheel。使用材料包中的sam2.1_hiera_small.pt，默认CPU即可运行。

独立任务可以给launch.py加参数：

```bash
--project-profile "/path/to/project_profile.json" --sam-checkpoint "/path/to/sam2.1_hiera_small.pt" --device cpu
```

## 加载和保存

下载个人任务目录与同级Shared。进入Manual模式打开任务video.mp4，task.json自动加载配置、标注和权重路径。已有reviewed.json时恢复它；保存默认写入reviewed.json。编号、FPS和帧数必须匹配。

独立视频用Load HOI annotations选择配套稿。导入旧格式时，工具先将原文件放入archived，再转换为当前格式。

选event和实例，拖动或缩放当前框，用Track Correction选择前向/后向及帧数。检查建议后应用，Undo可撤回。Ctrl＋滚轮缩放；右键删除当前框。Frame review用于核验和跳转未完成帧。项目标签和具体标注要求随内部材料发放。

## 常见问题

安装和启动使用同一环境的Python。缺少SAM模块时补装后端；权重路径指向本地现有文件。Linux缺Qt库时按英文说明安装系统依赖。Qt插件报错时用launch.py启动，检查是否混用了系统Qt、Conda与venv。测试范围见[VALIDATION](../VALIDATION.md)。
