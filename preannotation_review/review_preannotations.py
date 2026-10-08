"""CPU-only review entry; for the IMPACT desktop tool."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform.startswith('linux') and Path('/etc/fonts/fonts.conf').is_file():
    os.environ.setdefault('FONTCONFIG_FILE', '/etc/fonts/fonts.conf')
    os.environ.setdefault('FONTCONFIG_PATH', '/etc/fonts')

# Import cv2 before fixing plugin paths: wheels may inject a different Qt path.
import cv2
cv2.setNumThreads(1)
os.environ.pop('QT_QPA_FONTDIR', None)
import PyQt5
from PyQt5.QtCore import QLibraryInfo
_root = Path(QLibraryInfo.location(QLibraryInfo.PluginsPath))
for _candidate in (Path(sys.prefix) / 'Library/plugins',
                   Path(sys.prefix) / 'plugins',
                   Path(PyQt5.__file__).parent / 'Qt5/plugins'):
    if (_candidate / 'platforms').is_dir():
        _root = _candidate
        break
if (_root / 'platforms').is_dir():
    os.environ['QT_PLUGIN_PATH'] = str(_root)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(_root / 'platforms')

from PyQt5.QtWidgets import QApplication
from review_gui import ReviewWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('IMPACT 预标注审核')
    window = ReviewWindow()
    window.show()
    if len(sys.argv) > 1:
        window.open_path(sys.argv[1])
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
