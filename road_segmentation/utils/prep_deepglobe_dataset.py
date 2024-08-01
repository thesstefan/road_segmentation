import os
from PIL import Image
import numpy as np

def split_image(dataset_dir, output_dir, image_id):
    # Construct file paths for the image and its corresponding mask
    img_path = os.path.join(dataset_dir, f'{image_id}_sat.jpg')
    mask_path = os.path.join(dataset_dir, f'{image_id}_mask.png')
    
    # Load the image and mask
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Define the size of the small images
    sub_size = 512
    width, height = img.size
    
    index = 0
    for i in range(0, height, sub_size):
        for j in range(0, width, sub_size):
            # Crop the image and mask
            img_cropped = img.crop((j, i, j + sub_size, i + sub_size))
            mask_cropped = mask.crop((j, i, j + sub_size, i + sub_size))
            
            # Convert mask to array to check if it's empty
            mask_array = np.array(mask_cropped)
            if np.any(mask_array):
                # Save the cropped images if the mask is not empty
                img_cropped.save(os.path.join(output_dir, 'images', f'{image_id}_{index}.png'))
                mask_cropped.save(os.path.join(output_dir, 'labels', f'{image_id}_{index}.png'))
                index += 1

def process_all_images(dataset_dir, output_dir):
    # List all files in the dataset directory that are images
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('_sat.jpg')]
    
    # Process each file
    for file in image_files:
        image_id = file[:-8]  # Remove the '_sat.jpg' suffix to get the image ID
        split_image(dataset_dir, output_dir, image_id)

# Example usage:
dataset_dir = '/Users/jeffreyzweidler/Desktop/Local_CIL/resources/datasets/DeepGlobe/train'
output_dir = '/Users/jeffreyzweidler/Desktop/Local_CIL/resources/datasets/DeepGlobe/mod_train'

process_all_images(dataset_dir, output_dir)
