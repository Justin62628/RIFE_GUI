import sys
import QCandyUi
from PyQt5.QtWidgets import *
import win32
import win32gui
try:
    import RIFE_GUI_Utils.RIFE_GUI_Backend as RIFE_GUI_Backend
except ImportError as e:
    print("Import Utils Error")
    import RIFE_GUI_Backend

app = QApplication(sys.argv)
form = QCandyUi.CandyWindow.createWindow(RIFE_GUI_Backend.RIFE_GUI_BACKEND(), theme="blueDeep", ico_path="icon.ico", title="RIFE GUI v6.1.0")
form.show()
app.exec_()
