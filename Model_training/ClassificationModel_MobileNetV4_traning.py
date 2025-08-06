import os
import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset, random_split
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score ,classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

#Data_loader
# Constants
CLASS_NAMES = ['healthy_soyabean', 'mosaic', 'rust'] #Classes to be classified
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
DATASET_DIR = r"Path_Dataset"

# Dataset class
class LeafDataset(Dataset):
    def __init__(self, root_dir, img_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # Converts to [0,1] and (C,H,W)
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cls_name in CLASS_NAMES:
            cls_path = os.path.join(self.root_dir, cls_name)
            label = CLASS_TO_IDX[cls_name]
            for file in os.listdir(cls_path):
                if file.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
                    img_path = os.path.join(cls_path, file)
                    samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

# Split data function
def split_dataset(dataset, ratio=0.8):
    total_size = len(dataset)
    train_size = int(ratio * total_size)
    test_size = total_size - train_size
    return random_split(dataset, [train_size, test_size])



#Utils_forGraph_Parameter_Saving_modelSaving
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    # Plot Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss Curve')
    os.makedirs("trained_models", exist_ok=True)
    plt.savefig(f"trained_models/{model_name}_loss.png")
    plt.close()

    # Plot Accuracy Curve
    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.legend()
    plt.title(f'{model_name} - Accuracy Curve')
    plt.savefig(f"trained_models/{model_name}_acc.png")
    plt.close()

def save_confusion_matrix(y_true, y_pred, model_name, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    os.makedirs("trained_models", exist_ok=True)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"trained_models/{model_name}_cm.png")
    plt.close()

def log_metrics(model_name, batch_size, epochs, accuracy, precision, recall, f1_score):
    log_path = "model_results_rotate_zoom_median.csv"
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Batch Size", "Epochs", "Accuracy", "Precision", "Recall", "F1 Score"])
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name, batch_size, epochs,
            round(accuracy, 4), round(precision, 4),
            round(recall, 4), round(f1_score, 4)
        ])

#Model_taining_Starts

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in self.base.parameters():
            param.requires_grad = False

        self.base.classifier = nn.Identity()  # Remove default classifier
        #classifier V2
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
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

def train_mobilenetv2_model(img_size=(224, 224), batch_size=16, epochs=15, patience=3):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = LeafDataset(root_dir=DATASET_DIR,transform=transform)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MobileNetV2FeatureExtractor(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.00015)

    best_loss = np.inf
    wait = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"MobileNetV2_{timestamp}"
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), f"trained_models/{model_name}.pth")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name)
    save_confusion_matrix(y_true, y_pred, model_name, CLASS_NAMES)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    log_metrics("MobileNetV2_classifier_V2_lr0.00015", batch_size, epoch+1, acc, prec, rec, f1)

if __name__ == "__main__":
    train_mobilenetv2_model()
