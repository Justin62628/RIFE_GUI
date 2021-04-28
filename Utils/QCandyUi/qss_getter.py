# const color string
WHITE = "#FFFFFF"
# WHITE = "#CCCCCC"
BLACK = "#000000"
RED = "#FF0000"
GREEN = "#00FF00"
BLUE = "#0000FF"
PURPLE = "#B23AEE"
WATCHET = "#1C86EE"
LIGHTGREEN = "#ECFEFE"
BLUEGREEN = "#33CCCC"
DEEPBLUEGREEN = "#015F5F"
DARKBLUEGREEN = "#28AAAA"
GRAY = "#999999"
LIGHTGRAY = "#CCCCCC"

DEEPBLUE = "#131820"
HARDBLUE = "#1D2839"
LIGHTBLUE = "2DA0F7"
BLUEGRAY = "#313C4F"
LIGHTGRAY = "#777E88"
DEEPBROWN = "#85743D"
LIGHTBROWN = "F2C94C"

fontLight = "#FFFFFF",
fontDark = "#192e4b"
normal = "#4c73a8"
light = "#88b5f1"
deep = "#2e486c"
disLight = "#CCCCCC"
disDark = "#999999"

ArrowURL = ""

qss = f"""
QObject{{color:{WHITE}}}

QPushButton{{padding:0px;border-radius:5px;color:{LIGHTBLUE};background:{LIGHTBLUE};border:2px solid {LIGHTBLUE};}}
QPushButton:hover, pressed{{color:{light};background:{fontDark};}}
QPushButton:disabled{{color:{disDark};background:{disLight};}}

QProgressBar{{text-align: center; border:2px solid {HARDBLUE}; border-radius:5px; selection-color:{WHITE};selection-background-color:{light};}}

QLineEdit{{border-style:none;padding:2px;border-radius:5px;border:2px solid {LIGHTBLUE};selection-color:{WHITE};selection-background-color:{LIGHTGRAY};}}
QLineEdit:focus{{border:2px solid {fontDark};}}
QLineEdit:disabled{{color:{LIGHTGRAY};}}

QPlainTextEdit{{border-style:none;padding:2px;border-radius:10px;border:2px solid {LIGHTBLUE};font-family:宋体;selection-color:{WHITE};selection-background-color:{fontDark}}}
QPlainTextEdit:focus{{border:2px solid {fontDark};}}

QGroupBox {{
    /* 背景渐变色
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #E0E0E0, stop: 1 #FFFFFF);*/

    /* 边框 */
    border: 2px solid gray;

    /* 倒角 */
    border-radius: 5px;

    /* 就像墙上挂着的两个相框,margin指的是相框与相框的距离
       padding指的是每个相框里照片与相框边框的距离 */
    margin-top: 20px;
    padding-top: 10px;
    font-size: 18px;
    
}}

/* 标题设置 */
QGroupBox::title {{
    /* 位置 */
    subcontrol-origin: margin;
    subcontrol-position: top left;
    /* 内边框,上下和左右 */
    padding: 10px 15px;
}}

QTextBrowser{{border-style:none;padding:2px;border-radius:5px;border:2px solid {LIGHTBLUE};selection-color:{fontLight};selection-background-color:{fontDark}}}
QTextBrowser:focus{{border:2px solid {fontDark};}}
QTextEdit{{padding:2px;border-radius:5px;border:2px solid {LIGHTBLUE};selection-color:%s;selection-background-color:{fontDark}}}
QTextEdit:focus{{border:2px solid {fontDark};}} 
QTextEdit:hover{{border:2px solid {light};}}

QListWidget,QScrollArea{{border-style:none;padding:2px;border-radius:5px;border:2px solid {LIGHTGRAY};selection-color:{WHITE};selection-background-color:{GRAY}}}

QComboBox{{background:%s;padding:2px;border-radius:5px;border:2px solid %s;}}
QComboBox:focus{{border:2px solid %s;}}
QComboBox:on{{border:2px solid %s;}}
QComboBox:disabled{{color:%s;}}
QComboBox::drop-down{{border-style:solid;border-radius:5px;}}
QComboBox QAbstractItemView{{border:2px solid %s;background:transparent;selection-background-color:%s;}}
QComboBox::down-arrow{{image:url(%s); border:20px;}}
"""

spin_qss = f"""
QSpinBox {{
    padding:2px;border-radius:5px;border:2px solid {LIGHTGRAY};background:transparent;
}}
QSpinBox::up-arrow {{
    image: url(%s);border:20px;
}}
QSpinBox::down-arrow {{
    image: url(%s);border:20px;
}}
QSpinBox::hover {{
    border:2px solid %s;
}}
QSpinBox::disabled{{color:{LIGHTGRAY}; border-color: {LIGHTGRAY}}
"""
qss += spin_qss
qss += spin_qss.replace("QSpinBox", "QDoubleSpinBox")
qss += spin_qss.replace("QSpinBox", "QTimeEdit")

qss += f"""
QProgressBar{{font:9pt;height:9px;background:{DEEPBLUE};border-radius:2px;text-align:center;border:1px solid {LIGHTBLUE};}}
QProgressBar:chunk{{border-radius:2px;background-color:{GREEN};margin:2px}}

QSlider::groove:horizontal,QSlider::add-page:horizontal{{height:2px;border-radius:2px;background:{LIGHTGRAY};}}
QSlider::sub-page:horizontal{{height:2px;border-radius:2px;background:{GRAY};}}
QSlider::handle:horizontal{{width:2px;margin-top:-2px;margin-bottom:-2px;border-radius:2px;
                            background:qradialgradient(spread:pad,cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,stop:0.6 #FFFFFF,stop:0.8 {GRAY});}}

QRadioButton::indicator:unchecked{{image: url(%s);}}
QRadioButton::indicator:checked{{image: url(%s);}}
QRadioButton::indicator:checked:hover{{image: url(%s);}}
QRadioButton::disabled{{color: {LIGHTGRAY}}}

QCheckBox::indicator:unchecked{{image: url(%s);}}
QCheckBox::indicator:checked{{image: url(%s);}}
QCheckBox::indicator:unchecked:hover{{image: url(%s);}}
QCheckBox::indicator:checked:hover{{image: url(%s);}}
QCheckBox::disabled, QCheckBox::indicator::disabled{{color: {LIGHTGRAY}}}

QScrollBar::handle{{background:{GRAY};border-radius:5px;min-height:10px}}
QScrollBar::handle:pressed{{background:{GRAY}}}
QScrollBar:horizontal {{
            height:15px;
        }}
        QScrollBar:vertical {{
            width:15px;
        }}
"""

"""
        QScrollBar::up-arrow:vertical{{
            image: url({up_arrow_url});border:20px;
         }}
         QScrollBar::down-arrow:vertical{{
            image: url({arrowimageurl});border:20px;
         }}
         QScrollBar::left-arrow:horizontal{{
            image: url({left_arrow_url});border:20px;
         }}
         QScrollBar::right-arrow:horizontal{{
            image: url({right_arrow_url});border:20px;
         }}
         QScrollBar {{
            border:5px {handlebarcolor};
        }}
                 /* 添加右侧控制块 */
        QScrollBar::add-line:horizontal, QScrollBar::add-line:vertical, QScrollBar::sub-line:horizontal, QScrollBar::sub-line:vertical{{
            /* 外边框 */
            border:2px {handlebarcolor};
            
            /* 背景色 */
            background:{handlebarcolor};
            
            /* 宽度 */
            width:4px;

        }}
"""

# def getPushButtonQss(normalColor, normalTextColor, hoverColor, hoverTextColor, pressedColor, pressedTextColor,
#                      disableColor, disableTextColor):
#     str2 = "QPushButton:hover{{color:%s;background:%s;}}" % (hoverTextColor, hoverColor)
#     str3 = "QPushButton:pressed{{color:%s;background:%s;}}" % (pressedTextColor, pressedColor)
#     str4 = "QPushButton:disabled{{color:%s;background:%s;}}" % (disableTextColor, disableColor)
#     str5 = "QProgressBar{{text-align: center; border:2px solid %s; border-radius:5px; selection-color:%s;selection-background-color:%s;}}" % (
#         normalColor, WHITE, hoverColor)
#     return str1 + str2 + str3 + str4 + str5


# def getLineeditQss(normalColor, focusColor):
#     str1 = "QLineEdit{{border-style:none;padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s;}}" % (
#         normalColor, WHITE, focusColor)
#     str2 = "QLineEdit:focus{{border:2px solid %s;}}" % (focusColor)
#     str3 = "QLineEdit:disabled{{color:%s;}}" % (LIGHTGRAY)
#     return str1 + str2 + str3


# def getPlaineditQss(normalColor, focusColor):
#     str1 = "QPlainTextEdit{{border-style:none;padding:2px;border-radius:10px;border:2px solid %s;font-family:宋体;selection-color:%s;selection-background-color:%s}}" % (
#         normalColor, WHITE, focusColor)
#     str2 = "QPlainTextEdit:focus{{border:2px solid %s;}}" % (focusColor)
#
#     str3 = """
#             QGroupBox {{
#                 /* 背景渐变色
#                 background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                                   stop: 0 #E0E0E0, stop: 1 #FFFFFF);*/
#
#                 /* 边框 */
#                 border: 2px solid gray;
#
#                 /* 倒角 */
#                 border-radius: 5px;
#
#                 /* 就像墙上挂着的两个相框,margin指的是相框与相框的距离
#                    padding指的是每个相框里照片与相框边框的距离 */
#                 margin-top: 20px;
#                 padding-top: 10px;
#                 font-size: 18px;
#
#             }}
#
#             /* 标题设置 */
#             QGroupBox::title {{
#                 /* 位置 */
#                 subcontrol-origin: margin;
#                 subcontrol-position: top left;
#                 /* 内边框,上下和左右 */
#                 padding: 10px 15px;
#
#             }}
#
#     """
#     return str1 + str2 + str3


# def getTextBrowerQss(normalColor, focusColor):
#     str1 = "QTextBrowser{{border-style:none;padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s}}" % (
#         normalColor, WHITE, focusColor)
#     str2 = "QTextBrowser:focus{{border:2px solid %s;}}" % (focusColor)
#     str3 = "QTextEdit{{padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s}}" % (
#         normalColor, WHITE, focusColor)
#     str4 = "QTextEdit:focus{{border:2px solid %s;}} QTextEdit:hover{{border:2px solid %s;}}" % (focusColor, focusColor)
#     str5 = "QListWidget,QScrollArea{{border-style:none;padding:2px;border-radius:5px;border:2px solid %s;selection-color:%s;selection-background-color:%s}}" % (
#         normalColor, WHITE, focusColor)
#     return str1 + str2 + str3 + str4 + str5


# def getComboxQss(backgroundColor, normalColor, focusColor, arrowimageurl):
#     up_arrow_url = arrowimageurl.replace("down", "up")
#
#     str1 = "QComboBox{{background:%s;padding:2px;border-radius:5px;border:2px solid %s;}}" % (
#         backgroundColor, normalColor)
#     # str1 = "QComboBox{{background:transparent;padding:2px;border-radius:5px;border:2px solid %s;}}" % (
#     #     normalColor)
#     str2 = "QComboBox:focus{{border:2px solid %s;}}" % (focusColor)
#     str3 = "QComboBox:on{{border:2px solid %s;}}" % (focusColor)
#     str4 = "QComboBox:disabled{{color:%s;}}" % (LIGHTGRAY)
#     str5 = "QComboBox::drop-down{{border-style:solid;border-radius:5px;}}"
#     str6 = "QComboBox QAbstractItemView{{border:2px solid %s;background:transparent;selection-background-color:%s;}}" % (
#         focusColor, focusColor)
#     str7 = "QComboBox::down-arrow{{image:url(%s); border:20px;}}" % (arrowimageurl)
#     str8 = """
#             QSpinBox {{
#                 padding:2px;border-radius:5px;border:2px solid %s;background:transparent;
#             }}
#
#             QSpinBox::up-arrow {{
#                 image: url(%s);border:20px;
#             }}
#             QSpinBox::down-arrow {{
#                 image: url(%s);border:20px;
#             }}
#
#             QSpinBox::hover {{
#                 border:2px solid %s;
#             }}
#
#             QSpinBox::disabled{{color:%s; border-color: %s}}
#
#     """ % (focusColor, up_arrow_url, arrowimageurl, focusColor, LIGHTGRAY, LIGHTGRAY)
#     str9 = str8.replace("QSpinBox", "QDoubleSpinBox")
#     str10 = str8.replace("QSpinBox", "QTimeEdit")
#     return str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9 + str10


# def getProgressBarQss(normalColor, chunkColor):
#     barHeight = str(8)
#     barRadius = str(8)
#     str1 = "QProgressBar{{font:9pt;height:%spx;background:%s;border-radius:%spx;text-align:center;border:1px solid %s;}}" % (
#         barHeight, normalColor, barRadius, normalColor)
#     str2 = "QProgressBar:chunk{{border-radius:%spx;background-color:%s;margin:2px}}" % (barRadius, chunkColor)
#     return str1 + str2


# def getSliderQss(normalColor, grooveColor, handleColor):
#     sliderHeight = str(8)
#     sliderRadius = str(4)
#     handleWidth = str(13)
#     handleRadius = str(6)
#     handleOffset = str(3)
#     str1 = "QSlider::groove:horizontal,QSlider::add-page:horizontal{{height:%spx;border-radius:%spx;background:%s;}}" % (
#         sliderHeight, sliderRadius, normalColor)
#     str2 = "QSlider::sub-page:horizontal{{height:%spx;border-radius:%spx;background:%s;}}" % (
#         sliderHeight, sliderRadius, grooveColor)
#     str3 = "QSlider::handle:horizontal{{width:%spx;margin-top:-%spx;margin-bottom:-%spx;border-radius:%spx;" \
#            "background:qradialgradient(spread:pad,cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,stop:0.6 #FFFFFF,stop:0.8 %s);}}" % (
#                handleWidth, handleOffset, handleOffset, handleRadius, handleColor)
#     return str1 + str2 + str3
#
#
# def getRadioButtonQss(normimageurl, downimageurl, normimageurlhover, downimageurlhover):
#     # str1 = "QRadioButton::indicator{{width:15px;height:15px;}}"
#     str1 = ""
#     str2 = "QRadioButton::indicator:unchecked{{image: url(%s);}}" % (normimageurl)
#     str3 = "QRadioButton::indicator:checked{{image: url(%s);}}" % (downimageurl)
#     str4 = "QRadioButton::indicator:checked:hover{{image: url(%s);}}" % (downimageurlhover)
#     str5 = "QRadioButton::disabled{{color: %s}}" % (LIGHTGRAY)
#     return str1 + str2 + str3 + str4 + str5
#
#
# def getCheckBoxQss(normimageurl, checkimageurl, normimageurlhover, checkimageurlhover):
#     # str1 = "QCheckBox::indicator{{width:15px;height:15px;}}"
#     str1 = ""
#     str2 = "QCheckBox::indicator:unchecked{{image: url(%s);}}" % (normimageurl)
#     str3 = "QCheckBox::indicator:checked{{image: url(%s);}}" % (checkimageurl)
#     str4 = "QCheckBox::indicator:unchecked:hover{{image: url(%s);}}" % (normimageurlhover)
#     str5 = "QCheckBox::indicator:checked:hover{{image: url(%s);}}" % (checkimageurlhover)
#     str6 = "QCheckBox::disabled, QCheckBox::indicator::disabled{{color: %s}}" % (LIGHTGRAY)
#
#     return str1 + str2 + str3 + str4 + str5 + str6
#
#
# def getTabWidgetQss(normalTabColor, normalTabTextColor, tabBorderColor):
#     str1 = "QTabWidget{{color:%s; background:%s;font-size: 40px;}}" % (normalTabTextColor, normalTabColor)
#     str2 = "QTabWidget::tab-bar{{left:8px; width: 100px;}}"
#     str3 = "QTabBar::tab{{color:%s; background:%s;border-top:0px solid %s;" \
#            "border-top-left-radius: 6px; border-top-right-radius: 6px;padding: 2px; min-width: 8ex;}}" % (
#                normalTabTextColor, normalTabColor, tabBorderColor)
#     str4 = "QTabBar::tab:hover{{color:%s; background:%s;border:4px;}}" % (normalTabColor, normalTabTextColor)
#     str5 = """
#               QTabBar::tab:selected{{color:%s; background:%s;border-top-left-radius: 6px; border-top-right-radius: 6px;border:2px solid #C2C7CB; border-bottom: 1px;border-bottom-color: #C2C7CB;}}
#               QTabBar::tab:!selected {{
#               margin-top: 3px; /* make non-selected tabs look smaller */}}
#
#               QTabWidget::pane {{ /* The tab widget frame */
#                     padding: 2px;
#                     border: 2px solid #C2C7CB;
#                     border-radius: 4px solid #C2C7CB;
#                 }}
#
#                 QTabWidget::tab-bar {{
#                     left: 5px; /* move to the right by 5px */
#                 }}
#
#                 /* make use of negative margins for overlapping tabs */
#                 QTabBar::tab:selected {{
#                     /* expand/overlap to the left and right by 4px */
#                     margin-left: -3px;
#                     margin-right: -3px;
#                 }}
#
#                 QTabBar::tab:first:selected {{
#                     margin-left: 0; /* the first selected tab has nothing to overlap with on the left */
#                 }}
#
#                 QTabBar::tab:last:selected {{
#                     margin-right: 0; /* the last selected tab has nothing to overlap with on the right */
#                 }}
#
#                 QTabBar::tab:only-one {{
#                     margin: 0; /* if there is only one tab, we don't want overlapping margins */
#                 }}
#
#           """ % (normalTabColor, normalTabTextColor,)
#     return str1 + str3 + str4 + str5


# def getScrollbarQss(handlebarcolor, arrowimageurl):
    left_arrow_url = arrowimageurl.replace("down", "left")
    right_arrow_url = arrowimageurl.replace("down", "right")
    up_arrow_url = arrowimageurl.replace("down", "up")
    # str1 = "QScrollBar{{background:transparent;width:10px;padding-top:11px;padding-bottom:11px}}"
    # str1 = "QScrollBar{{background:transparent;padding-top:11px;padding-bottom:11px}}"
    str2 = "QScrollBar::handle{{background:%s;border-radius:5px;min-height:10px}}" % (handlebarcolor)
    str3 = "QScrollBar::handle:pressed{{background:%s}}" % (GRAY)
    str4 = f"""
        QScrollBar:horizontal {{
            height:15px;
        }}
        QScrollBar:vertical {{
            width:15px;
        }}
        QScrollBar::up-arrow:vertical{{
            image: url({up_arrow_url});border:20px;
         }}
         QScrollBar::down-arrow:vertical{{
            image: url({arrowimageurl});border:20px;
         }}
         QScrollBar::left-arrow:horizontal{{
            image: url({left_arrow_url});border:20px;
         }}
         QScrollBar::right-arrow:horizontal{{
            image: url({right_arrow_url});border:20px;
         }}
         QScrollBar {{
            border:5px {handlebarcolor};
        }}
                 /* 添加右侧控制块 */
        QScrollBar::add-line:horizontal, QScrollBar::add-line:vertical, QScrollBar::sub-line:horizontal, QScrollBar::sub-line:vertical{{
            /* 外边框 */
            border:2px {handlebarcolor};

            /* 背景色 */
            background:{handlebarcolor};

            /* 宽度 */
            width:4px;

        }}

           """

    return str2 + str3 + str4
