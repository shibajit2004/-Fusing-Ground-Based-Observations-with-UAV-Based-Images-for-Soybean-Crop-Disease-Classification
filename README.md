# -Fusing-Ground-Based-Observations-with-UAV-Based-Images-for-Soybean-Crop-Disease-Classification
This system uses UAV and ground-level images to detect soybean leaf diseases early. U-Net++ segments leaves from drone images, while a MobileNetV4-based model classifies them as Healthy, Mosaic, or Rust. Color-coded boxes show results, aiding scalable, real-time, and cost-effective crop health monitoring.
## 🚀 Features

- 🌱 **Segment** leaves from drone images using U-Net++  
- 🔍 **Classify** leaf health using a MobileNetV2-based PyTorch model  
- 🎨 **Color-coded bounding boxes** for disease types:  
  - Green: Healthy  
  - Red: Rust  
  - Yellow: Mosaic  
- 🖼️ **Drag & Drop / File Upload** support  
- 🔍 **Zoom & Pan** in processed results  
- 🧠 Powered by deep learning (Keras + PyTorch)

---

## 📁 Folder Structure

```
MainApp/
│
├── main.py                    # Main application entry point
├── image_processor.py         # Contains segmentation & classification logic
├── upload_page.py             # UI code for upload interface
├── result_page.py             # UI code for result display
├── upload_page.ui             # Qt Designer .ui file for upload page
├── result_page.ui             # Qt Designer .ui file for result page
├── logo_app.png               # App icon/logo
├── banner.webp                # Optional UI banner
├── MobileNetV2_20250728_121209_new.pth  # PyTorch classifier model
├── unetpp_verysmall.h5        # Keras segmentation model
├── result_page.jpg            # Sample processed image (optional)
└── README.md
```

---

## 🛠️ Requirements

Install these via pip:

```bash
pip install -r requirements.txt
```

### `requirements.txt` example:

```
torch
torchvision
opencv-python
numpy
tensorflow
PySide6
```

---

## 💡 How to Use

### Option 1: Run via Python
```bash
python main.py
```

### Option 2: Run as Executable
- Use the provided Inno Setup script to build an installer.
- After installation, run `AgriVision` from your Start Menu or Desktop shortcut.

---

## 📦 Packaging & Installer

To create an executable and installer:
1. **Compile with PyInstaller:**

   ```bash
   pyinstaller --noconfirm --onefile --windowed main.py
   ```

2. **Build Installer with Inno Setup:**
   Use the `.iss` file provided and compile it with Inno Setup Compiler.

---

## ⚙️ Models Used

- **Segmentation (U-Net++)**: Trained on aerial leaf masks (`.h5` Keras model)
- **Classification (MobileNetV2)**: PyTorch model for leaf-level disease classification

---

## 🖼️ GUI Highlights

- ✅ Drag-and-drop or browse to select an image  
- 🔄 Automatically segments and classifies leaves  
- 🖼️ Shows both original and processed image  
- 🔎 Zoom and pan functionality for detailed inspection  

---

## 📌 Notes

- Processing may take a few seconds depending on image resolution and system performance.
- If GPU is available, PyTorch will use it automatically.

---

## 🧑‍💻 Author

Developed by Souvik De , Shibajit Chatterjee And Antarik Manna
