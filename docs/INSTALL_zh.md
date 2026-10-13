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

下载自己的姓名目录和同级Shared，保留原目录结构。启动工具，通过文件夹按钮或菜单Load Video...打开姓名/tasks/reassembly或disassembly/视频编号/video.mp4。相邻的task.json自动加载Shared中的项目配置与annotations.json，并登记SAM/YOLO权重路径；无需逐个导入。已有reviewed.json时优先恢复它，保存默认写入reviewed.json。工具核对视频编号、FPS和帧数。

不要把video.mp4单独移出任务目录。review.json是原始模型审核材料，主编辑器读取task.json指定的编辑稿。正常重启会恢复上次工程和播放位置；发现更新的自动恢复稿时，按提示选择恢复。音频审核独立运行：打开个人音频任务目录内的review.html。

独立视频先通过⋯ → Import → Project Profile...加载提供的project_profile.json，再用HOI Annotations...选择配套稿。导入旧格式时，工具先将原文件放入archived，再转换为当前格式。

选event和实例，拖动或缩放参考框，再用Track Correction勾选目标、设置当前帧两侧的区间。点击OK后任务在后台排队，可以继续标注。小进度条显示已处理的目标帧数；Details可取消、重试或恢复保存的结果。结果自动回填为未核验框，用播放和Frame review检查，Undo可撤回一次回填。传播期间的人工修改会保留，新参考框会截断旧结果并触发后续传播。

任务和部分结果保存到视频旁的.impact-recovery目录，重新打开同一视频会恢复中断任务；未完成的分支从参考帧重新计算。请保证视频目录可写。默认CPU最多使用4个线程，较慢电脑可设置IMPACT_SAM2_CPU_THREADS=1；后端空闲两分钟后释放模型。Ctrl＋滚轮缩放；右键删除当前框。Keyframes菜单用于跳转缺少必需框的关键帧。项目标签和具体标注要求随内部材料发放。

## 常见问题

安装和启动使用同一环境的Python。缺少SAM模块时补装后端；权重路径指向本地现有文件。Linux缺Qt库时按英文说明安装系统依赖。Qt插件报错时用launch.py启动，检查是否混用了系统Qt、Conda与venv。测试范围见[VALIDATION](../VALIDATION.md)。
