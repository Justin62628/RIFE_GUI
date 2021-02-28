import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

try:
    import RIFE_GUI_Utils.RIFE_GUI_Backend as RIFE_GUI_Backend
except ImportError as e:
    import RIFE_GUI_Backend

app = QApplication(sys.argv)
form = RIFE_GUI_Backend.RIFE_GUI_BACKEND()
form.show()
app.exec_()
