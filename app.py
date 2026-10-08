import argparse
import os
import sys

_QT_DLL_HANDLES = []


def _bootstrap_qt_runtime() -> None:
    """
    Force Qt to use the PyQt5-bundled runtime/plugins.
    This avoids PATH/plugin conflicts in mixed environments (e.g. Anaconda + other Qt installs).
    """
    try:
        import PyQt5  # delay QtWidgets import until plugin paths are normalized
    except Exception:
        return
    pyqt_root = os.path.dirname(PyQt5.__file__)
    qt_root = os.path.join(pyqt_root, "Qt5")
    pyqt_plugin_root = os.path.join(qt_root, "plugins")
    pyqt_platform_root = os.path.join(pyqt_plugin_root, "platforms")
    pyqt_bin = os.path.join(qt_root, "bin")

    # Prefer Conda Qt runtime when available; it is usually ABI-matched with the environment.
    conda_root = sys.prefix
    conda_plugin_root = os.path.join(conda_root, "Library", "plugins")
    if not os.path.isdir(os.path.join(conda_plugin_root, "platforms")):
        unix_plugins = os.path.join(conda_root, "plugins")
        if os.path.isdir(os.path.join(unix_plugins, "platforms")):
            conda_plugin_root = unix_plugins
    conda_platform_root = os.path.join(conda_plugin_root, "platforms")
    conda_bin = os.path.join(conda_root, "Library", "bin")

    if os.path.isdir(conda_platform_root):
        plugin_root = conda_plugin_root
        platform_root = conda_platform_root
        qt_bin = conda_bin
    else:
        plugin_root = pyqt_plugin_root
        platform_root = pyqt_platform_root
        qt_bin = pyqt_bin

    # Clear externally injected plugin paths that often point to incompatible Qt builds.
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    if os.path.isdir(plugin_root):
        os.environ["QT_PLUGIN_PATH"] = plugin_root
    if os.path.isdir(platform_root):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_root
    if os.path.isdir(qt_bin):
        cur_path = os.environ.get("PATH", "")
        parts = [p.strip() for p in cur_path.split(os.pathsep) if p.strip()]
        norm_parts = {os.path.normcase(os.path.normpath(p)) for p in parts}
        norm_qt_bin = os.path.normcase(os.path.normpath(qt_bin))
        if norm_qt_bin not in norm_parts:
            os.environ["PATH"] = qt_bin + os.pathsep + cur_path
        # Python 3.8+ on Windows: ensure dependent DLL lookup includes Qt bin.
        if hasattr(os, "add_dll_directory"):
            try:
                _QT_DLL_HANDLES.append(os.add_dll_directory(qt_bin))
            except Exception:
                pass


_bootstrap_qt_runtime()

from utils.feature_env import load_feature_env_defaults  # noqa: E402
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
load_feature_env_defaults(repo_root=_REPO_ROOT)
# Some torch kernels need this before torch-dependent modules load.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QIcon, QFont, QFontDatabase  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from utils.op_logger import OperationLogger  # noqa: E402

# OpenCV wheels can override Qt plugin paths during the imports above.
_bootstrap_qt_runtime()


def _set_windows_app_id(app_id: str) -> None:
    if os.name != "nt" or not app_id:
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def _configure_app_font(app: QApplication) -> None:
    try:
        db = QFontDatabase()
        families = set(db.families())
    except Exception:
        families = set()
    preferred = ["Segoe UI", "Microsoft YaHei UI", "Tahoma", "Arial"]
    font = app.font()
    for family in preferred:
        if not families or family in families:
            font.setFamily(family)
            break
    font.setStyleHint(QFont.SansSerif)
    target_pt = 8.0 if os.name == "nt" else 8.75
    font.setPointSizeF(target_pt)
    app.setFont(font)

def _resolve_app_icon() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "icon.ico"),
        os.path.join(root, "icon.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oplog",
        action="store_true",
        help="Enable operation logging (writes alongside annotation file).",
    )
    args, _ = parser.parse_known_args()

    _set_windows_app_id("cvhci.video.annotation.impact_hoi")
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass
    try:
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    _configure_app_font(app)
    icon_path = _resolve_app_icon()
    if icon_path:
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
    logger = OperationLogger(enabled=bool(args.oplog))
    w = MainWindow(logger=logger)
    if icon_path:
        w.setWindowIcon(QIcon(icon_path))
    w.show()
    sys.exit(app.exec_())


# Subtitle conversion feature is not implemented
# Bbox Genera1tor feature is not implemented
# Segmantaion Assistant feature is not implemented
# Interactive Segmentation feature is not implemented
