import re
import os
import json
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

try:
    from RIFE_GUI_Utils import RIFE_GUI
except ImportError as e:
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
    interp_start = 0
    interp_cnt = 0
    chunk = 0
    bitrate = 0
    preset = ""
    crop = ""
    resize = ""
    CRF = 5

    HWACCEL = True
    UHD = False
    PAUSE = False
    DEBUG = False
    scene_only = False
    extract_only = False
    quick_extract = False
    RIFE_only = False
    render_only = False
    concat_only = False

    fp16 = False
    reverse = False
    interp_scale = 1.0
    Scedet = 2
    ScedetT = 0.2


class RIFE_Run_Other_Threads(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, command, parent=None):
        super(RIFE_Run_Other_Threads, self).__init__(parent)
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        self.command = command

    def run(self):
        print(f"[CMD Thread]: Start execute {self.command}")
        os.system(self.command)
        pass

    pass


class RIFE_Run_Thread(QThread):
    run_signal = pyqtSignal(str)

    def __init__(self, user_args, parent=None):
        self.RIFE_args = RIFE_Args()
        super(RIFE_Run_Thread, self).__init__(parent)
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        if user_args is not None:
            self.RIFE_args = user_args
        #: Indent above
        self.command = ""
        self.build_command()

    def fillQuotation(self, string):
        if string[0] != '"':
            return f'"{string}"'

    def build_command(self):
        if os.path.splitext(self.RIFE_args.OneLineShotPath)[-1] == ".exe":
            self.command = self.RIFE_args.OneLineShotPath + " "
        else:
            self.command = f"python {self.RIFE_args.OneLineShotPath} "
        self.command += f'-i {self.fillQuotation(self.RIFE_args.InputFileName)} '
        if os.path.isfile(self.RIFE_args.OutputFolder):
            print("[Thread]: OutputPath with FileName detected")
            self.RIFE_args.OutputFolder = os.path.dirname(self.RIFE_args.OutputFolder)
        self.command += f'--output {self.RIFE_args.OutputFolder}/concat_all.mp4 '
        self.command += f'--rife {self.fillQuotation(self.RIFE_args.RIFEPath)} '
        self.command += f'--ffmpeg {self.fillQuotation(self.RIFE_args.FFmpeg)} '
        self.command += f'--fps {self.RIFE_args.InputFPS} '
        self.command += f'--ratio {int(self.RIFE_args.Exp)} '
        self.command += f'--crf {self.RIFE_args.CRF} '
        self.command += f'--scale {self.RIFE_args.interp_scale} '
        if self.RIFE_args.interp_start:
            self.command += f'--interp-start {self.RIFE_args.interp_start} '
        if self.RIFE_args.interp_cnt:
            self.command += f'--interp-cnt {self.RIFE_args.interp_cnt} '
        if self.RIFE_args.chunk:
            self.command += f'--chunk {self.RIFE_args.chunk} '
        if self.RIFE_args.crop not in ["0", ""]:
            if self.RIFE_args.UHD:
                self.command += f'--UHD-crop {self.RIFE_args.crop} '
            else:
                self.command += f'--HD-crop {self.RIFE_args.crop} '
        if self.RIFE_args.resize:
            self.command += f'--resize {self.RIFE_args.resize} '
        if self.RIFE_args.bitrate:
            self.command += f'--bitrate {self.RIFE_args.bitrate}M '
        if self.RIFE_args.preset:
            self.command += f'--preset {self.RIFE_args.preset} '
        if self.RIFE_args.Scedet:
            self.command += f'--scdet {self.RIFE_args.Scedet} '
        if float(self.RIFE_args.ScedetT):
            self.command += f'--scdet-threshold {self.RIFE_args.ScedetT} '

        if self.RIFE_args.reverse:
            self.command += f'--reverse '
        if self.RIFE_args.UHD:
            self.command += f'--UHD '
        if self.RIFE_args.HWACCEL:
            self.command += f'--hwaccel '
        if self.RIFE_args.quick_extract:
            self.command += f'--quick-extract '
        if self.RIFE_args.scene_only:
            self.command += f'--scene-only '
        elif self.RIFE_args.extract_only:
            self.command += f'--extract-only '
        elif self.RIFE_args.RIFE_only:
            self.command += f'--rife-only '
        elif self.RIFE_args.render_only:
            self.command += f'--render-only '
        elif self.RIFE_args.concat_only:
            self.command += f'--concat-only '
        if self.RIFE_args.fp16:
            self.command += f'--fp16 '
        if self.RIFE_args.PAUSE:
            self.command += f'--pause '
        if self.RIFE_args.DEBUG:
            self.command += f'--debug '
        print("[Thread]: Designed Command:")
        print(self.command)

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

    def select_file(self, filename, folder=False):
        if folder:
            directory = QFileDialog.getExistingDirectory(None, caption="选取文件夹")
            return directory
        directory = QFileDialog.getOpenFileName(None, caption=f"选择{filename}")
        return directory[0]

    def quick_concat(self):
        input_v = self.ConcatInputV.text()
        input_a = self.ConcatInputA.text()
        output_v = self.OutputConcat.text()
        self.load_current_settings()
        offset_args = self.ConcatOffsetSelector.value()
        if offset_args:
            offset_args = f"-itsoffset {offset_args}"
        else:
            offset_args = ""
        if not output_v or not input_a or not output_v:
            reply = QMessageBox.warning(self,
                                        "Parameters unfilled",
                                        "请填写输入或输出视频路径！",
                                        QMessageBox.Yes)
            return
        ffmpeg = os.path.join(self.RIFE_args.FFmpeg, "ffmpeg.exe")
        ffmpeg_command = f"""
            {ffmpeg} -i {input_a} -i {input_v} -map 1:v:0 -map 0:a:0 -c copy -shortest {output_v} -y
        """.strip().strip("\n").replace("\n", "")
        print(f"ffmpeg_command for concat {ffmpeg_command}")
        # thrRIFE_Run_Other_Threads(ffmpeg_command)
        # thread.start()
        os.system(ffmpeg_command)

    def auto_set(self):
        chunk_list = list()
        if not len(self.OutputFolder.toPlainText()):
            print("OutputFolder path is empty, pls enter it first")
            reply = QMessageBox.warning(self,
                                        "Parameters unfilled",
                                        "请移步Stage1填写输出文件夹！",
                                        QMessageBox.Yes)
            return
        for f in os.listdir(self.OutputFolder.toPlainText()):
            if re.match("chunk-[\d+].*?\.mp4", f):
                chunk_list.append(f)
        if not len(chunk_list):
            self.StartFrame.setText("0")
            self.StartCntFrame.setText("1")
            self.StartChunk.setText("1")
            print("AutoSet Ready")
            return
        print("Found Previous Chunks")
        chunk_list.sort(key=lambda x: int(x.split('-')[2]))
        last_chunk = chunk_list[-1]
        match_result = re.findall("chunk-(\d+)-(\d+)-(\d+)\.mp4", last_chunk)[0]

        chunk = int(match_result[0])
        first_frame = int(match_result[1])
        last_frame = int(match_result[2])
        first_interp_cnt = (last_frame + 1) * (2 ** self.RIFE_args.Exp) + 1
        self.StartChunk.setText(str(chunk))
        self.StartFrame.setText(str(first_frame))
        self.StartCntFrame.setText(str(first_interp_cnt))
        pass

    @pyqtSlot(bool)
    def on_InputBrowser_clicked(self):
        input_filename = self.select_file('视频文件')
        self.InputFileName.setText(input_filename)

    @pyqtSlot(bool)
    def on_OutputBrowser_clicked(self):
        output_folder = self.select_file('输出项目文件夹', folder=True)
        self.OutputFolder.setText(output_folder)

    @pyqtSlot(bool)
    def on_FFmpegBrowser_clicked(self):
        FFmpeg = self.select_file('FFmpeg.exe文件夹所在路径', folder=True)
        self.FFmpegPath.setText(FFmpeg)

    @pyqtSlot(bool)
    def on_RIFEBrowser_clicked(self):
        rife_path = self.select_file('inference_img_only.exe路径')
        self.RIFEPath.setText(rife_path)

    @pyqtSlot(bool)
    def on_OneLineShotBrowser_clicked(self):
        onelineshot_path = self.select_file('OneLineShot路径')
        self.OneLineShotPath.setText(onelineshot_path)

    @pyqtSlot(bool)
    def on_AutoSet_clicked(self):
        self.auto_set()

    @pyqtSlot(bool)
    def on_ConcatButton_clicked(self):
        self.quick_concat()
        # self.auto_set()
        pass

    @pyqtSlot(str)
    def on_ExpSelecter_currentTextChanged(self, currentExp):
        input_fps = self.InputFPS.text()
        selected_exp = currentExp[1:]
        if input_fps:
            self.OutputFPSReminder.setText(f"预计输出帧率：{float(input_fps) * float(selected_exp)}")

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, tabIndex):
        if tabIndex == 2:
            """Step 3"""
            print("[Main]: Start Loading Settings")
            self.load_current_settings()

    def save_current_settings(self):
        with open("../Settings.json", 'w', encoding='utf-8') as w:
            settings = dict()
            settings["InputFile"] = self.RIFE_args.InputFileName
            settings["OutputFolder"] = self.RIFE_args.OutputFolder
            settings["FFmpeg"] = self.RIFE_args.FFmpeg
            settings["RIFEPath"] = self.RIFE_args.RIFEPath
            settings["OneLineShotPath"] = self.RIFE_args.OneLineShotPath
            settings["InputFPS"] = self.RIFE_args.InputFPS
            settings["Exp"] = self.RIFE_args.Exp
            settings["CRF"] = self.RIFE_args.CRF
            settings["interp_start"] = self.RIFE_args.interp_start
            settings["interp_cnt"] = self.RIFE_args.interp_cnt
            settings["chunk"] = self.RIFE_args.chunk
            settings["resize"] = self.RIFE_args.resize

            settings["UHD"] = self.RIFE_args.UHD
            settings["HWACCEL"] = self.RIFE_args.HWACCEL
            settings["PAUSE"] = self.RIFE_args.PAUSE
            settings["DEBUG"] = self.RIFE_args.DEBUG
            settings["scene_only"] = self.RIFE_args.scene_only
            settings["extract_only"] = self.RIFE_args.extract_only
            settings["quick_extract"] = self.RIFE_args.quick_extract
            settings["RIFE_only"] = self.RIFE_args.RIFE_only
            settings["render_only"] = self.RIFE_args.render_only
            settings["concat_only"] = self.RIFE_args.concat_only
            settings["reverse"] = self.RIFE_args.reverse
            settings["fp16"] = self.RIFE_args.fp16
            settings["interp_scale"] = self.RIFE_args.interp_scale
            settings["bitrate"] = self.RIFE_args.bitrate
            settings["preset"] = self.RIFE_args.preset
            settings["crop"] = self.RIFE_args.crop
            settings["Scedet"] = self.RIFE_args.Scedet
            settings["ScedetT"] = self.RIFE_args.ScedetT

            json.dump(settings, w)
            print("[Main]: Save Current Settings")

    def load_json_settings(self):
        settings_path = "../Settings.json"
        if not os.path.exists(settings_path):
            print("[Main]: No Previous Settings Found, Return")
            return
        with open("../Settings.json", 'r', encoding='utf-8') as r:
            settings = json.load(r, )
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
            self.StartFrame.setText(settings["interp_start"])
            self.StartCntFrame.setText(settings["interp_cnt"])
            self.StartChunk.setText(settings["chunk"])
            self.BitrateSelector.setValue(float(settings["bitrate"]))
            # self.PresetSelector.setCurrentText(settings["preset"])
            self.CropSettings.setText(settings["crop"])
            self.ResizeSettings.setText(settings["resize"])
            self.ScdetSelector.setValue(int(settings["Scedet"]))
            self.ScdetTSelector.setValue(float(settings["ScedetT"]))

    def load_current_settings(self):
        self.RIFE_args.InputFileName = self.InputFileName.toPlainText()
        self.RIFE_args.OutputFolder = self.OutputFolder.toPlainText()
        self.RIFE_args.FFmpeg = self.FFmpegPath.toPlainText()
        self.RIFE_args.RIFEPath = self.RIFEPath.toPlainText()
        self.RIFE_args.OneLineShotPath = self.OneLineShotPath.toPlainText()
        self.RIFE_args.InputFPS = float(self.InputFPS.text()) if len(self.InputFPS.text()) else 0
        self.RIFE_args.Exp = math.log(int(self.ExpSelecter.currentText()[1:]), 2)
        self.RIFE_args.bitrate = float(self.BitrateSelector.value())
        self.RIFE_args.preset = self.PresetSelector.currentText().split('[')[0]
        self.RIFE_args.crop = self.CropSettings.text()
        self.RIFE_args.resize = self.ResizeSettings.text()
        self.RIFE_args.interp_scale = float(self.InterpScaleSelector.currentText())
        self.RIFE_args.Scedet = self.ScdetSelector.value()
        self.RIFE_args.ScedetT = self.ScdetTSelector.text()

        self.RIFE_args.CRF = int(self.CRFSelector.value())
        self.RIFE_args.interp_start = int(self.StartFrame.text())
        self.RIFE_args.interp_cnt = int(self.StartCntFrame.text())
        self.RIFE_args.chunk = int(self.StartChunk.text())

        self.RIFE_args.UHD = bool(self.UHDChecker.isChecked())
        self.RIFE_args.HWACCEL = bool(self.HwaccelChecker.isChecked())
        self.RIFE_args.PAUSE = bool(self.PauseChecker.isChecked())
        self.RIFE_args.DEBUG = bool(self.DebugChecker.isChecked())
        self.RIFE_args.scene_only = bool(self.ExtractScenesOnlyChecker.isChecked())
        self.RIFE_args.extract_only = bool(self.ExtractOnlyChecker.isChecked())
        self.RIFE_args.quick_extract = bool(self.QuickExtractChecker.isChecked())
        self.RIFE_args.RIFE_only = bool(self.RIFEOnlyChecker.isChecked())
        self.RIFE_args.render_only = bool(self.RenderOnlyChecker.isChecked())
        self.RIFE_args.concat_only = bool(self.ConcatOnlyChecker.isChecked())
        self.RIFE_args.fp16 = bool(self.FP16Checker.isChecked())
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
    app = QApplication(sys.argv)
    form = RIFE_GUI_BACKEND()
    form.show()
    app.exec_()
