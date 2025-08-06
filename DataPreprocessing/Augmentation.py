import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Paths
source_dir = r'Path_input_directory'
output_dir = r'Path_output_directory'

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Number of new augmented images to add per class
AUGMENT_COUNT = 1648

# Custom augmentation function
def custom_augment(img_array):
    choice = random.choice(['rotate', 'zoom', 'both'])

    if choice == 'rotate':
        angle = random.uniform(-40, 40)
        img_array = tf.keras.preprocessing.image.random_rotation(
            img_array, angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
        )
    
    elif choice == 'zoom':
        zx = random.uniform(1.0, 1.2)
        zy = random.uniform(1.0, 1.2)
        img_array = tf.keras.preprocessing.image.random_zoom(
            img_array, (zx, zy), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
        )

    elif choice == 'both':
        angle = random.uniform(-40, 40)
        img_array = tf.keras.preprocessing.image.random_rotation(
            img_array, angle, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
        )
        zx = random.uniform(1.0, 1.2)
        zy = random.uniform(1.0, 1.2)
        img_array = tf.keras.preprocessing.image.random_zoom(
            img_array, (zx, zy), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
        )

    return img_array

# Process each subfolder
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    save_folder = os.path.join(output_dir, folder)
    os.makedirs(save_folder, exist_ok=True)

    images = [img for img in os.listdir(folder_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
    print(f"\n Processing: {folder} — Found {len(images)} source images")
    
    print(f"Generating {AUGMENT_COUNT} new augmented images...")
    count = 0
    while count < AUGMENT_COUNT:
        img_name = random.choice(images)
        img_path = os.path.join(folder_path, img_name)
        try:
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            aug_x = custom_augment(x[0])
            save_name = f"aug_{count}.jpg"
            array_to_img(aug_x).save(os.path.join(save_folder, save_name))
            count += 1
        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")

    print(f" Added {AUGMENT_COUNT} augmented images to '{folder}'")

print("\n Done! All folders augmented.")
