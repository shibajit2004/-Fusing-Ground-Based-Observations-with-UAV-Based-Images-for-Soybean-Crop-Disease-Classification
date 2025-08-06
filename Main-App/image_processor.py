# image_processor.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.models import load_model
from torchvision import transforms, models

# ====== CONFIGURATION ======
SEG_MODEL_PATH = r"unetpp_verysmall.h5"
CLASSIFIER_MODEL_PATH = r"MobileNetV2_20250728_121209_new.pth"
CLASS_NAMES = ['healthy_soyabean', 'mosaic', 'rust']
IMAGE_SIZE = 256
CLASSIFIER_INPUT_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== MODEL DEFINITIONS ======
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.base.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

# ====== Load Models Once ======
print("Loading models...")
segmentation_model = load_model(SEG_MODEL_PATH)
classifier_model = MobileNetV2FeatureExtractor(num_classes=len(CLASS_NAMES))
classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
classifier_model.to(device)
classifier_model.eval()

# ====== Classifier Transform ======
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def process_image_with_models(frame):
    original_height, original_width = frame.shape[:2]
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # SEGMENTATION
    pred = segmentation_model.predict(input_img)[0, :, :, 0]
    binary_mask = (pred > 0.25).astype(np.uint8)
    binary_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w < 10 or h < 10) or (w > 500 or h > 500):
            continue

        crop = frame[y:y + h, x:x + w]

        try:
            input_tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = classifier_model(input_tensor)
                prob = F.softmax(logits, dim=1)
                class_id = torch.argmax(prob, dim=1).item()
                disease_name = CLASS_NAMES[class_id]
        except Exception as e:
            disease_name = "Error"
            print(f"Classification failed for region {(x, y, w, h)}: {e}")
            disease_name = "Error"

        # Assign box color
        if disease_name == "healthy_soyabean":
            color = (0, 255, 0)  # Green
            # --- Healthy leaf boxed here ---
        elif disease_name == "rust":
            color = (0, 0, 255)  # Red
        elif disease_name == "mosaic":
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # orange for error or unknown

        # Draw only the colored rectangle (no text)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)

    return frame  # processed image

