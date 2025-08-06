import os
import cv2
import numpy as np
from glob import glob

def normalize_color_channels(img):
    """Normalize each channel to have zero mean and unit variance, then scale back to 0–255"""
    img = img.astype(np.float32)
    norm_img = np.zeros_like(img)
    
    for c in range(3):  # B, G, R
        channel = img[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel)
        if std < 1e-6: std = 1e-6  # prevent divide by zero
        norm_channel = (channel - mean) / std

        # Scale back to 0–255 for saving
        norm_channel = cv2.normalize(norm_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm_img[:, :, c] = norm_channel

    return norm_img.astype(np.uint8)

def resize_and_normalize_images(input_dir, output_dir, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, '*.*'))

    if not image_paths:
        print("❌ No images found in input directory.")
        return

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        resized_img = cv2.resize(img, target_size)
        normalized_img = normalize_color_channels(resized_img)

        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, normalized_img)

    print(f"✅ Resized + normalized images saved to: {output_dir}")

# Example usage
if __name__ == "__main__":
    input_folder = r"path_input_folder_directory"       # change to your input folder
    output_folder = r"path_output_folder_directory"    # change to your desired output folder
    resize_and_normalize_images(input_folder, output_folder)

