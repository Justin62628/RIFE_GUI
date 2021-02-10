import re
import os
import json
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import RIFE_GUI
import sys
MAC = True
try:
    from PyQt5.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False


class RIFE_Args:
    InputFileName = ""
    OutputFolder = ""
    FFmpeg = ""
    RIFEPath = ""
    OneLineShotPath = ""

    InputFPS = 23.976
    Exp = 1
    CRF = 5
    start = 0
    chunk = 0
    end = 0
    render_round = 1
    bitrate = 0
    preset = ""
    crop = ""

    UHD = False
    HWACCEL = True
    PAUSE = False
    RIFE_only = False
    render_only = False

    accurate = False
    reverse = False


class RIFE_Run_Thread(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, user_args, parent=None):
        super(RIFE_Run_Thread, self).__init__(parent)
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        self.RIFE_args = user_args
        self.command = ""
        self.build_command()

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def build_command(self):
        self.command = self.RIFE_args.OneLineShotPath + " "

        self.command += f'-i {self.fillQuotation(self.RIFE_args.InputFileName)} '
        if '.' in self.RIFE_args.OutputFolder.split('/')[-1].split('.')[-1]:
            print("[Thread]: OutputPath with FileName detected")
            self.RIFE_args.OutputFolder = os.path.basename(self.RIFE_args.OutputFolder)
        self.command += f'--output {self.RIFE_args.OutputFolder}/output.mp4 '
        self.command += f'--rife {self.fillQuotation(self.RIFE_args.RIFEPath)} '
        self.command += f'--ffmpeg {self.fillQuotation(self.RIFE_args.FFmpeg)} '
        self.command += f'--fps {self.RIFE_args.InputFPS} '
        self.command += f'--ratio {int(self.RIFE_args.Exp)} '
        self.command += f'--crf {self.RIFE_args.CRF} '
        self.command += f'--start {self.RIFE_args.start} '
        if self.RIFE_args.end:
            self.command += f'--end {self.RIFE_args.end} '
        if self.RIFE_args.chunk:
            self.command += f'--chunk {self.RIFE_args.chunk} '
        if self.RIFE_args.render_round:
            self.command += f'--round {self.RIFE_args.round} '
        if self.RIFE_args.crop not in ["0", ""]:
            if self.RIFE_args.UHD:
                self.command += f'--UHDcrop {self.RIFE_args.crop} '
            else:
                self.command += f'--HDcrop {self.RIFE_args.crop} '
        if self.RIFE_args.bitrate:
            self.command += f'--bitrate {self.RIFE_args.bitrate}M '
        if self.RIFE_args.preset:
            self.command += f'--preset {self.RIFE_args.preset} '
        if self.RIFE_args.reverse:
            self.command += f'--reverse '
        if self.RIFE_args.UHD:
            self.command += f'--UHD '
        if self.RIFE_args.accurate:
            self.command += f'--accurate '
        if self.RIFE_args.HWACCEL:
            self.command += f'--hwaccel '
        if self.RIFE_args.render_only:
            self.command += f'--render_only '
        if self.RIFE_args.RIFE_only:
            self.command += f'--rife_only '
        if self.RIFE_args.PAUSE:
            self.command += f'--pause '
        print("[Thread]: Designed Command:")
        print(self.command)
        # TODO: Resize

    def run(self):
        print("[Thread]: Start")
        os.system(self.command)
        pass

    pass


class RIFE_GUI_BACKEND(QDialog, RIFE_GUI.Ui_RIFEDialog):
    found = pyqtSignal(int)
    notfound = pyqtSignal(int)

    def __init__(self, parent=None):
        super(RIFE_GUI_BACKEND, self).__init__(parent)
        self.setupUi(self)
        self.RIFE_args = RIFE_Args()
        self.thread = None
        self.load_json_settings()
        # if not MAC:
        #     self.findButton.setFocusPolicy(Qt.NoFocus)
        """
        1. select file, return string
        2. input box content address
        3. tab select, update info at selecting tab 3
        4. os.system
        
        1. 输入文件分别四个响应
        2. 其他参数输入
        3. 点击Step3返回参数以及建立线程
        4. 点击即渲染
        5. 保存设置
        """

    def select_file(self, filename):
        directory = QFileDialog.getOpenFileName(None, caption=f"选择{filename}")
        return directory[0]

    @pyqtSlot(bool)
    def on_InputBrowser_clicked(self):
        input_filename = self.select_file('视频文件')
        self.InputFileName.setText(input_filename)

    @pyqtSlot(bool)
    def on_OutputBrowser_clicked(self):
        output_folder = self.select_file('输出项目文件夹')
        self.OutputFolder.setText(os.path.dirname(output_folder))

    @pyqtSlot(bool)
    def on_FFmpegBrowser_clicked(self):
        FFmpeg = self.select_file('FFmpeg.exe路径')
        self.FFmpegPath.setText(FFmpeg)

    @pyqtSlot(bool)
    def on_RIFEBrowser_clicked(self):
        rife_path = self.select_file('inference_img_only.py路径')
        self.RIFEPath.setText(rife_path)

    @pyqtSlot(bool)
    def on_OneLineShotBrowser_clicked(self):
        onelineshot_path = self.select_file('OneLineShot路径')
        self.OneLineShotPath.setText(onelineshot_path)

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self, currentExp):
        input_fps = self.InputFPS.text()
        selected_exp = currentExp[1:]
        if input_fps:
            self.OutputFPSReminder.setText(f"预计输出帧率：{round(float(input_fps) * float(selected_exp))}")

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tabIndex):
        if tabIndex == 2:
            """Step 3"""
            print("[Main]: Start Loading Settings")
            self.load_current_settings()

    def save_current_settings(self):
        with open("Settings.json", 'w', encoding='utf-8') as w:
            settings = dict()
            settings["InputFile"] = self.RIFE_args.InputFileName
            settings["OutputFolder"] = self.RIFE_args.OutputFolder
            settings["FFmpeg"] = self.RIFE_args.FFmpeg
            settings["RIFEPath"] = self.RIFE_args.RIFEPath
            settings["OneLineShotPath"] = self.RIFE_args.OneLineShotPath
            settings["InputFPS"] = self.RIFE_args.InputFPS
            settings["Exp"] = self.RIFE_args.Exp
            settings["CRF"] = self.RIFE_args.CRF
            settings["start"] = self.RIFE_args.start
            settings["end"] = self.RIFE_args.end
            settings["chunk"] = self.RIFE_args.chunk
            settings["render_round"] = self.RIFE_args.render_round

            settings["UHD"] = self.RIFE_args.UHD
            settings["HWACCEL"] = self.RIFE_args.HWACCEL
            settings["PAUSE"] = self.RIFE_args.PAUSE
            settings["RIFE_only"] = self.RIFE_args.RIFE_only
            settings["render_only"] = self.RIFE_args.render_only
            settings["accurate"] = self.RIFE_args.accurate
            settings["reverse"] = self.RIFE_args.reverse
            settings["bitrate"] = self.RIFE_args.bitrate
            settings["preset"] = self.RIFE_args.preset
            settings["crop"] = self.RIFE_args.crop

            json.dump(settings, w)
            print("[Main]: Save Current Settings")

    def load_json_settings(self):
        settings_path = "Settings.json"
        if not os.path.exists(settings_path):
            print("[Main]: No Previous Settings Found, Return")
            return
        with open("Settings.json", 'r', encoding='utf-8') as r:
            settings = json.load(r,)
            for s in settings:
                settings[s] = str(settings[s])
            self.InputFileName.setText(settings["InputFile"])
            self.OutputFolder.setText(settings["OutputFolder"])
            self.FFmpegPath.setText(settings["FFmpeg"])
            self.RIFEPath.setText(settings["RIFEPath"])
            self.OneLineShotPath.setText(settings["OneLineShotPath"])
            self.InputFPS.setText(settings["InputFPS"])
            self.ExpSelecter.setCurrentText(settings["Exp"])
            self.CRFSelector.setValue(int(settings["CRF"]))
            self.StartFrame.setText(settings["start"])
            self.EndFrame.setText(settings["end"])
            self.StartChunk.setText(settings["chunk"])
            self.Round.setText(settings["render_round"])
            self.BitrateSelector.setValue(float(settings["bitrate"]))
            # self.PresetSelector.setCurrentText(settings["preset"])
            self.CropSettings.setText(settings["crop"])

            if bool(settings["UHD"]):
                self.UHDChecker.setChecked(True)
            if bool(settings["HWACCEL"]):
                self.HwaccelChecker.setChecked(True)
            if bool(settings["PAUSE"]):
                self.PauseChecker.setChecked(True)
            if bool(settings["RIFE_only"]):
                self.RIFEOnlyChecker.setChecked(True)
            if bool(settings["render_only"]):
                self.RenderOnlyChecker.setChecked(True)
            if bool(settings["accurate"]):
                self.AccurateChecker.setChecked(True)
            if bool(settings["reverse"]):
                self.ReverseChecker.setChecked(True)

    def load_current_settings(self):
        self.RIFE_args.InputFileName = self.InputFileName.toPlainText()
        self.RIFE_args.OutputFolder = self.OutputFolder.toPlainText()
        self.RIFE_args.FFmpeg = self.FFmpegPath.toPlainText()
        self.RIFE_args.RIFEPath = self.RIFEPath.toPlainText()
        self.RIFE_args.OneLineShotPath = self.OneLineShotPath.toPlainText()
        self.RIFE_args.InputFPS = float(self.InputFPS.text())
        self.RIFE_args.Exp = math.log(int(self.ExpSelecter.currentText()[1:]), 2)
        self.RIFE_args.bitrate = float(self.BitrateSelector.value())
        self.RIFE_args.preset = self.PresetSelector.currentText().split('[')[0]
        self.RIFE_args.crop = self.CropSettings.text()

        self.RIFE_args.CRF = int(self.CRFSelector.value())
        self.RIFE_args.start = int(self.StartFrame.text())
        self.RIFE_args.chunk = int(self.StartChunk.text())
        self.RIFE_args.end = int(self.EndFrame.text())
        self.RIFE_args.render_round = int(self.Round.text())

        self.RIFE_args.UHD = bool(self.UHDChecker.isChecked())
        self.RIFE_args.HWACCEL = bool(self.HwaccelChecker.isChecked())
        self.RIFE_args.PAUSE = bool(self.PauseChecker.isChecked())
        self.RIFE_args.RIFE_only = bool(self.RIFEOnlyChecker.isChecked())
        self.RIFE_args.render_only = bool(self.RenderOnlyChecker.isChecked())
        self.RIFE_args.accurate = bool(self.AccurateChecker.isChecked())
        self.RIFE_args.reverse = bool(self.ReverseChecker.isChecked())
        print("[Main]: Download all settings")
        status_check = "[当前导出设置预览]\n\n"
        for name, value in vars(self.RIFE_args).items():
            status_check += f"{name} => {value}\n"
        self.OptionCheck.setText(status_check)
        self.OptionCheck.isReadOnly = True
        self.save_current_settings()
        pass

    @pyqtSlot(bool)
    def on_ProcessStart_clicked(self):
        self.thread = RIFE_Run_Thread(self.RIFE_args)
        self.thread.start()
        self.OptionCheck.setText("[一条龙启动，请移步命令行查看进度详情]")

    @pyqtSlot(bool)
    def on_CloseButton_clicked(self):
        self.save_current_settings()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    form = RIFE_GUI_BACKEND()
    form.show()
    app.exec_()
