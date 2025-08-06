# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'upload_page.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(800, 600)
        Form.setStyleSheet(u"\n"
"    QWidget {\n"
"        font-family: 'Segoe UI', sans-serif;\n"
"        background-color: #f5f5f5;\n"
"    }\n"
"    QLabel#label_title {\n"
"        font-size: 28px;\n"
"        font-weight: bold;\n"
"        color: #000000;\n"
"        background-color: transparent;\n"
"    }\n"
"    QLabel#logo_app {\n"
"        margin-right: 10px;\n"
"        background-color: transparent;\n"
"    }\n"
"    QLabel#logo_drop {\n"
"        background-color: transparent;\n"
"    }\n"
"    QFrame#drop_area {\n"
"        border: 2px dashed #66bb6a;\n"
"        border-radius: 16px;\n"
"        background-color: #e8f5e9;\n"
"        padding: 30px;\n"
"    }\n"
"    QFrame#drop_area:hover {\n"
"        background-color: #d0f0d3;\n"
"    }\n"
"    QLabel#label_drag {\n"
"        font-size: 20px;\n"
"        color: #000000;\n"
"        padding-top: 5px;\n"
"        background-color: transparent;\n"
"        \n"
"    }\n"
"    QPushButton#btn_select {\n"
"        background-color: #4CAF50;\n"
"        color: white;\n"
"      "
                        "  padding: 10px 24px;\n"
"        font-size: 14px;\n"
"        border: none;\n"
"        border-radius: 6px;\n"
"    }\n"
"    QPushButton#btn_select:hover {\n"
"        background-color: #45a049;\n"
"    }\n"
"    QLabel#label_footer {\n"
"        font-size: 12px;\n"
"        color: #FFFFFF;\n"
"        background-color: #000000;\n"
"        padding: 10px;\n"
"    }\n"
"    QWidget#headerWidget {\n"
"        background-color: #ffffff;\n"
"        padding: 10px;\n"
"    }\n"
"   ")
        self.mainLayout = QVBoxLayout(Form)
        self.mainLayout.setObjectName(u"mainLayout")
        self.headerWidget = QWidget(Form)
        self.headerWidget.setObjectName(u"headerWidget")
        self.headerLayout = QHBoxLayout(self.headerWidget)
        self.headerLayout.setObjectName(u"headerLayout")
        self.headerLayout.setContentsMargins(0, 0, 0, 0)
        self.spacer_left = QSpacerItem(20, 50, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.headerLayout.addItem(self.spacer_left)

        self.centerHeaderLayout = QHBoxLayout()
        self.centerHeaderLayout.setObjectName(u"centerHeaderLayout")
        self.logo_app = QLabel(self.headerWidget)
        self.logo_app.setObjectName(u"logo_app")
        self.logo_app.setPixmap(QPixmap(u"logo_app.png"))
        self.logo_app.setMaximumSize(QSize(50, 40))
        self.logo_app.setScaledContents(True)

        self.centerHeaderLayout.addWidget(self.logo_app)

        self.label_title = QLabel(self.headerWidget)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setAlignment(Qt.AlignVCenter|Qt.AlignLeft)

        self.centerHeaderLayout.addWidget(self.label_title)


        self.headerLayout.addLayout(self.centerHeaderLayout)

        self.spacer_right = QSpacerItem(20, 50, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.headerLayout.addItem(self.spacer_right)


        self.mainLayout.addWidget(self.headerWidget)

        self.label_banner = QLabel(Form)
        self.label_banner.setObjectName(u"label_banner")
        self.label_banner.setPixmap(QPixmap(u"banner.webp"))
        self.label_banner.setAlignment(Qt.AlignCenter)
        self.label_banner.setScaledContents(True)
        self.label_banner.setMaximumSize(QSize(16777215, 200))

        self.mainLayout.addWidget(self.label_banner)

        self.vertical_spacer_top = QSpacerItem(20, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.mainLayout.addItem(self.vertical_spacer_top)

        self.drop_area = QFrame(Form)
        self.drop_area.setObjectName(u"drop_area")
        self.dropLayout = QVBoxLayout(self.drop_area)
        self.dropLayout.setSpacing(20)
        self.dropLayout.setObjectName(u"dropLayout")
        self.dropLayout.setAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
        self.logo_drop = QLabel(self.drop_area)
        self.logo_drop.setObjectName(u"logo_drop")
        self.logo_drop.setPixmap(QPixmap(u"logo_dropbox.png"))
        self.logo_drop.setMaximumSize(QSize(300, 100))
        self.logo_drop.setAlignment(Qt.AlignCenter)
        self.logo_drop.setScaledContents(True)

        self.dropLayout.addWidget(self.logo_drop)

        self.label_drag = QLabel(self.drop_area)
        self.label_drag.setObjectName(u"label_drag")
        self.label_drag.setAlignment(Qt.AlignCenter)

        self.dropLayout.addWidget(self.label_drag)

        self.btn_select = QPushButton(self.drop_area)
        self.btn_select.setObjectName(u"btn_select")
        self.btn_select.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.dropLayout.addWidget(self.btn_select)


        self.mainLayout.addWidget(self.drop_area)

        self.vertical_spacer_bottom = QSpacerItem(20, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.mainLayout.addItem(self.vertical_spacer_bottom)

        self.label_footer = QLabel(Form)
        self.label_footer.setObjectName(u"label_footer")
        self.label_footer.setAlignment(Qt.AlignCenter)

        self.mainLayout.addWidget(self.label_footer)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        self.label_title.setText(QCoreApplication.translate("Form", u"AgriVision AI", None))
        self.label_drag.setText(QCoreApplication.translate("Form", u"Drag & Drop your image here", None))
        self.btn_select.setText(QCoreApplication.translate("Form", u"Select Image from Folder", None))
        self.label_footer.setText(QCoreApplication.translate("Form", u"\u00a9 2025 AgriVision AI - All rights reserved.", None))
        pass
    # retranslateUi

