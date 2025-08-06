# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'result_page.ui'
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
from PySide6.QtWidgets import (QApplication, QGraphicsView, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1024, 600)
        Form.setStyleSheet(u"\n"
"    QWidget {\n"
"        font-family: 'Segoe UI', sans-serif;\n"
"        background-color: #fafafa;\n"
"    }\n"
"    QLabel#label_original {\n"
"        font-size: 18px;\n"
"        font-weight: bold;\n"
"        color: #2e7d32;\n"
"    }\n"
"    QLabel#label_processed {\n"
"        font-size: 18px;\n"
"        font-weight: bold;\n"
"        color: #c62828;\n"
"    }\n"
"    QLabel#image_original {\n"
"        border: 1px solid #bbb;\n"
"        background-color: #ffffff;\n"
"        padding: 100px 10px;\n"
"        margin-bottom: 10px;\n"
"    }\n"
"    QPushButton#btn_back {\n"
"        background-color: #4CAF50;\n"
"        color: white;\n"
"        border: none;\n"
"        padding: 10px 20px;\n"
"        font-size: 20px;\n"
"        border-radius: 8px;\n"
"    }\n"
"    QPushButton#btn_back:hover {\n"
"        background-color: #43a047;\n"
"    }\n"
"    QGraphicsView#graphicsView_processed {\n"
"        background-color: #fff;\n"
"        border: 1px solid #bbb;\n"
"    }\n"
"   ")
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.leftLayout = QVBoxLayout()
        self.leftLayout.setObjectName(u"leftLayout")
        self.label_original = QLabel(Form)
        self.label_original.setObjectName(u"label_original")
        self.label_original.setMinimumSize(QSize(300, 100))
        self.label_original.setAlignment(Qt.AlignCenter)

        self.leftLayout.addWidget(self.label_original)

        self.image_original = QLabel(Form)
        self.image_original.setObjectName(u"image_original")
        self.image_original.setMinimumSize(QSize(300, 500))
        self.image_original.setAlignment(Qt.AlignCenter)
        self.image_original.setScaledContents(True)

        self.leftLayout.addWidget(self.image_original)

        self.btn_back = QPushButton(Form)
        self.btn_back.setObjectName(u"btn_back")
        self.btn_back.setMinimumSize(QSize(300, 100))

        self.leftLayout.addWidget(self.btn_back)


        self.horizontalLayout.addLayout(self.leftLayout)

        self.rightLayout = QVBoxLayout()
        self.rightLayout.setObjectName(u"rightLayout")
        self.label_processed = QLabel(Form)
        self.label_processed.setObjectName(u"label_processed")
        self.label_processed.setAlignment(Qt.AlignCenter)

        self.rightLayout.addWidget(self.label_processed)

        self.graphicsView_processed = QGraphicsView(Form)
        self.graphicsView_processed.setObjectName(u"graphicsView_processed")
        self.graphicsView_processed.setMinimumSize(QSize(600, 400))

        self.rightLayout.addWidget(self.graphicsView_processed)


        self.horizontalLayout.addLayout(self.rightLayout)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        self.label_original.setText(QCoreApplication.translate("Form", u"Inserted Image", None))
        self.btn_back.setText(QCoreApplication.translate("Form", u"\u2190 Back to Upload Page", None))
        self.label_processed.setText(QCoreApplication.translate("Form", u"Image after Disease Detection", None))
        pass
    # retranslateUi

