import os
import cv2
import random
from tqdm import tqdm
import shutil

# Settings
input_dir = r'D:\Shibajit Chatterjee\Cis Internship\preprocessed_dataset\rotate_zoom\leaf'  # Main folder with multiple subfolders
output_dir = r'D:\Shibajit Chatterjee\Cis Internship\preprocessed_dataset\rotate_zoom_median\leaf'
kernel_size = 5
filter_ratio = 0.4675345

# Prepare output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

#  Recursively collect image file paths
image_paths = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)

# Debug: How many images found
print(f" Found {len(image_paths)} images across all subfolders.")

# Randomly select subset to filter
num_to_filter = int(len(image_paths) * filter_ratio)
filtered_set = set(random.sample(image_paths, num_to_filter))

# Process and save all images
for img_path in tqdm(image_paths, desc="Processing images"):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f" Could not load image: {img_path}")
        continue

    # Construct relative path to preserve subfolder structure
    rel_path = os.path.relpath(img_path, input_dir)
    save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"{'median_' if img_path in filtered_set else 'original_'}{os.path.basename(img_path)}"
    save_path = os.path.join(save_dir, save_name)

    # Apply median filter if selected
    if img_path in filtered_set:
        img = cv2.medianBlur(img, kernel_size)

    # Save the image
    cv2.imwrite(save_path, img)

# Summary
print(f"\n All images processed.")
print(f" Total images: {len(image_paths)}")
print(f" Filtered with median: {num_to_filter}")
print(f"Output saved in: {output_dir}")
