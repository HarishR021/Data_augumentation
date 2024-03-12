# data_augmentation_script.py

import os
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa

# Set your input and output directories
input_directory = "C:\\Users\\lenovo\\Downloads\\dataugm\\folderL"
output_directory = r"C:\\Users\\lenovo\\Downloads\\dataugm\\augresfolL"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define augmentation sequence
augmentation_sequence = iaa.Sequential([
    iaa.Affine(rotate=(-40, 40)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
])

# Loop through each image in the input directory and generate augmented images
for filename in os.listdir(input_directory):
    img = Image.open(os.path.join(input_directory, filename))
    x = np.array(img)

    # Ensure the image is in RGB mode
    if x.shape[2] == 4:  # Check if it's RGBA
        x = x[:, :, :3]  # Keep only RGB channels

    # Apply augmentation
    augmented_images = augmentation_sequence(images=[x])

    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        Image.fromarray(augmented_image).save(os.path.join(output_directory, f"{filename.split('.')[0]}_aug_{i}.jpg"))