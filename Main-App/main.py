import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QStackedWidget, QFileDialog, QLabel, QGraphicsScene, QGraphicsPixmapItem, QProgressDialog,QGraphicsView)
from PySide6.QtGui import QPixmap , QIcon
from PySide6.QtCore import Qt, QTimer
from image_processor import process_image_with_models  # ADD THIS AT TOP
import cv2
from upload_page import Ui_Form as UploadUI
from result_page import Ui_Form as ResultUI


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale_factor = 1.15

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = self.scale_factor if zoom_in else 1 / self.scale_factor
        self.scale(factor, factor)


class UploadPage(QWidget):
    def __init__(self, stack, shared_data):
        super().__init__()
        self.ui = UploadUI()
        self.ui.setupUi(self)
        self.stack = stack
        self.shared_data = shared_data
        self.image_path = ""

        self.setAcceptDrops(True)
        if hasattr(self.ui, 'btn_next'):
            self.ui.btn_next.hide()

        self.ui.btn_select.clicked.connect(self.open_file_dialog)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for file_url in event.mimeData().urls():
            path = file_url.toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.set_image_path(path, source="Dropped")
                return
        self.ui.label_drag.setText("❌ Unsupported file format. Try PNG or JPG.")

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.set_image_path(path, source="Selected")

    def set_image_path(self, path, source):
        self.image_path = path
        filename = os.path.basename(path)
        self.ui.label_drag.setText(f"✅ Image {source}: {filename}")
        self.goto_result_with_delay()

    def goto_result_with_delay(self):
        if self.image_path:
            self.shared_data["image_path"] = self.image_path
            self.progress = QProgressDialog("Processing image...", None, 0, 0, self)
            self.progress.setWindowTitle("Please Wait")
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.setCancelButton(None)
            self.progress.setMinimumDuration(0)
            self.progress.show()
            QTimer.singleShot(100, self.complete_transition)

    def complete_transition(self):
        self.progress.close()
        self.stack.setCurrentIndex(1)


class ResultPage(QWidget):
    def __init__(self, stack, shared_data):
        super().__init__()
        self.ui = ResultUI()
        self.ui.setupUi(self)
        self.stack = stack
        self.shared_data = shared_data

        self.ui.btn_back.clicked.connect(self.go_back)

        layout = self.ui.graphicsView_processed.parentWidget().layout()
        self.graphics_view = ZoomableGraphicsView()
        layout.replaceWidget(self.ui.graphicsView_processed, self.graphics_view)
        self.ui.graphicsView_processed.deleteLater()

        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

    
    def showEvent(self, event):
        self.scene.clear()

        orig_path = self.shared_data.get("image_path", "")
        if not os.path.exists(orig_path):
            return

        # Load input image using OpenCV
        input_image = cv2.imread(orig_path)
        if input_image is None:
            return

        # Process image using the external segmentation+classification model
        processed_image = process_image_with_models(input_image)

        # Save for consistency
        processed_path = "process.jpg"
        cv2.imwrite(processed_path, processed_image)

        # Show original image
        pixmap = QPixmap(orig_path).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ui.image_original.setPixmap(pixmap)

        # Show processed image
        pixmap = QPixmap(processed_path)
        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.graphics_view.fitInView(item, Qt.KeepAspectRatio)
        
    def go_back(self):
        self.stack.setCurrentIndex(0)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgriVision AI")
        self.setWindowIcon(QIcon("logo_app.png"))  
        self.resize(1024, 600)

        self.stack = QStackedWidget()
        self.shared_data = {}

        self.upload_page = UploadPage(self.stack, self.shared_data)
        self.result_page = ResultPage(self.stack, self.shared_data)

        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.result_page)

        self.setCentralWidget(self.stack)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
