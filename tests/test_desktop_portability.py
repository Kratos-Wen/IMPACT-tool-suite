"""Native GUI and Unicode/spaced subprocess path smoke test; no models required."""
import os, sys, tempfile
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QApplication
from ui.hoi_window import HOIWindow
app = QApplication.instance() or QApplication([])
window = HOIWindow()
window.show()
app.processEvents()
with tempfile.TemporaryDirectory(prefix="IMPACT space 中文 ") as temporary:
    worker = Path(temporary) / "worker 中文.py"
    output = Path(temporary) / "result 中文.txt"
    worker.write_text("from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('OK', encoding='utf-8')\n", encoding="utf-8")
    process = QProcess()
    process.start(sys.executable, [str(worker), str(output)])
    assert process.waitForStarted(10000), process.errorString()
    assert process.waitForFinished(20000), process.errorString()
    assert process.exitCode() == 0, bytes(process.readAllStandardError())
    assert output.read_text(encoding="utf-8") == "OK"
window.hide()
print("DESKTOP_GUI_AND_UNICODE_SUBPROCESS_PASS", sys.platform)
