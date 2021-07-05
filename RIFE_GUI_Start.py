import sys
import traceback

import QCandyUi
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

"""High Resolution Support"""
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

try:
    from Utils import RIFE_GUI_Backend
except ImportError as e:
    traceback.print_exc()
    print("Not Find RIFE GUI Backend, please contact developers for support")
    input("Press Any Key to Quit")
    exit()

"""Release Version Control"""
version_free = False
version_tag = "3.3.2.alpha"
if "alpha" not in version_tag:
    if version_free:
        version_tag += " Community"
    else:
        version_tag += " Professional"
version_title = f"Squirrel Video Frame Interpolation {version_tag}"

"""Initiate APP"""
app = QApplication(sys.argv)
app_backend_module = RIFE_GUI_Backend
app_backend = app_backend_module.RIFE_GUI_BACKEND(free=version_free, version=version_tag)
try:
    form = QCandyUi.CandyWindow.createWindow(app_backend, theme="blueDeep", ico_path="svfi.png",
                                             title=version_title)
    form.show()
    app.exec_()
except Exception:
    app_backend_module.logger.critical(traceback.format_exc())
