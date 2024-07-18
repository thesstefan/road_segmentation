import os
from PIL import Image
import numpy as np

def split_image(input_dir, output_dir, image_name):
    # Load the image
    img_path = os.path.join(input_dir, 'images', f'{image_name}.tiff')
    img = Image.open(img_path)
    
    # Attempt to load the mask, checking both possible extensions
    mask_path_jpg = os.path.join(input_dir, 'labels', f'{image_name}.jpg')
    mask_path_tif = os.path.join(input_dir, 'labels', f'{image_name}.tif')
    
    if os.path.exists(mask_path_jpg):
        mask = Image.open(mask_path_jpg)
    elif os.path.exists(mask_path_tif):
        mask = Image.open(mask_path_tif)
    else:
        raise FileNotFoundError(f"No mask found for image {image_name} with .jpg or .tif extension.")
    
    # Ensure output directory exists
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Define the size of the small images
    sub_size = 750
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
                img_cropped.save(os.path.join(output_dir, 'images', f'{image_name}_{index}.png'))
                mask_cropped.save(os.path.join(output_dir, 'labels', f'{image_name}_{index}.png'))
                index += 1

def process_all_images(input_dir, output_dir):
    # List all files in the 'images' directory
    image_files = [f for f in os.listdir(os.path.join(input_dir, 'images')) if f.endswith('.tiff')]
    
    # Process each file
    for file in image_files:
        image_name = file[:-5]  # Remove the '.tiff' extension to get the image ID
        split_image(input_dir, output_dir, image_name)

# Example usage:
input_dir = '/Users/jeffreyzweidler/Desktop/Local_CIL/resources/datasets/Massachusetts/train'
output_dir = '/Users/jeffreyzweidler/Desktop/Local_CIL/resources/datasets/Massachusetts/mod_train'

process_all_images(input_dir, output_dir)

