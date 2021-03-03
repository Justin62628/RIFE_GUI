import sys

import QCandyUi
from PyQt5.QtWidgets import *

try:
    import RIFE_GUI_Utils.RIFE_GUI_Backend as RIFE_GUI_Backend
except ImportError as e:
    import RIFE_GUI_Backend

app = QApplication(sys.argv)
form = QCandyUi.CandyWindow.createWindow(RIFE_GUI_Backend.RIFE_GUI_BACKEND(), theme="pink")
form.show()
app.exec_()
