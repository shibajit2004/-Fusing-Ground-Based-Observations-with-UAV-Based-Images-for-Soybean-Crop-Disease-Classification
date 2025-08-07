import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU, BinaryAccuracy
from sklearn.model_selection import train_test_split

# Optional Dice coefficient
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # threshold
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    
# Enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Parameters
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
EPOCHS = 70
BATCH_SIZE = 8
N_CLASSES = 1

IMAGE_DIR = "./augmented_data/images"
MASK_DIR = "./augmented_data/masks"

# -------------------------------
# Data Loader
# -------------------------------
def load_data(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    images, masks = [], []
    for img_f, mask_f in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_f)
        mask_path = os.path.join(mask_dir, mask_f)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))

        img = img / 255.0
        mask = (mask / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

X, y = load_data(IMAGE_DIR, MASK_DIR)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# U-Net++ Model from Scratch
# -------------------------------
def conv_block(x, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x

def build_unetpp(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder
    x00 = conv_block(inputs, 4)
    x10 = MaxPooling2D(pool_size=(2, 2))(x00)
    x10 = conv_block(x10, 8)
    x20 = MaxPooling2D(pool_size=(2, 2))(x10)
    x20 = conv_block(x20, 16)
    x30 = MaxPooling2D(pool_size=(2, 2))(x20)
    x30 = conv_block(x30, 32)
    x40 = MaxPooling2D(pool_size=(2, 2))(x30)
    x40 = conv_block(x40, 64)

    # Decoder with dense skip connections
    x01 = conv_block(concatenate([UpSampling2D()(x10), x00]), 4)
    x11 = conv_block(concatenate([UpSampling2D()(x20), x10]), 8)
    x21 = conv_block(concatenate([UpSampling2D()(x30), x20]), 16)
    x31 = conv_block(concatenate([UpSampling2D()(x40), x30]), 32)

    x02 = conv_block(concatenate([UpSampling2D()(x11), x00, x01]), 4)
    x12 = conv_block(concatenate([UpSampling2D()(x21), x10, x11]), 8)
    x22 = conv_block(concatenate([UpSampling2D()(x31), x20, x21]), 16)

    x03 = conv_block(concatenate([UpSampling2D()(x12), x00, x01, x02]), 4)
    x13 = conv_block(concatenate([UpSampling2D()(x22), x10, x11, x12]), 8)

    x04 = conv_block(concatenate([UpSampling2D()(x13), x00, x01, x02, x03]), 4)

    outputs = Conv2D(1, 1, activation='sigmoid')(x04)

    return Model(inputs, outputs)

# -------------------------------
# Compile and Train
# -------------------------------
with strategy.scope():
    model = build_unetpp()
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy(name="bin_acc"), dice_coef]
    )

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2
)

# -------------------------------
# Evaluate and Save
# -------------------------------
loss, miou, acc = model.evaluate(X_val, y_val)
print(f"\n✅ U-Net++ Evaluation:\nLoss: {loss:.4f}, Mean IoU: {miou:.4f}, Accuracy: {acc:.4f}")

model.save("unetpp_verysmall.h5")
