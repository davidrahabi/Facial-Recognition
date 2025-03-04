import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

def data_aug(img):
    """Apply various augmentations to an image with stronger visible effects"""
    data = []
    
    # Convert to tensor once
    if not isinstance(img, tf.Tensor):
        original_img_tensor = tf.convert_to_tensor(img)
    else:
        original_img_tensor = img
        
    # Make sure the tensor has the right shape and type
    if len(original_img_tensor.shape) == 2:
        original_img_tensor = tf.expand_dims(original_img_tensor, axis=-1)
        original_img_tensor = tf.tile(original_img_tensor, [1, 1, 3])
    
    original_img_tensor = tf.cast(original_img_tensor, tf.uint8)
    
    # Create 9 distinct augmentations with more noticeable differences
    
    # 1. Increase brightness significantly
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_brightness(
        img_tensor, max_delta=0.2, seed=(1, 2))  # Increased from 0.02
    data.append(augmented)
    
    # 2. Decrease brightness (by adjusting in the opposite direction)
    img_tensor = tf.identity(original_img_tensor)
    # To darken: first convert to float, subtract brightness, then clip and convert back
    img_float = tf.cast(img_tensor, tf.float32)
    darkened = tf.clip_by_value(img_float - 40.0, 0, 255)  # Subtract constant value
    augmented = tf.cast(darkened, tf.uint8)
    data.append(augmented)
    
    # 3. Increase contrast significantly
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_contrast(
        img_tensor, lower=1.3, upper=1.8, seed=(3, 4))  # Much stronger contrast
    data.append(augmented)
    
    # 4. Decrease contrast
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_contrast(
        img_tensor, lower=0.4, upper=0.7, seed=(4, 5))  # Lower contrast
    data.append(augmented)
    
    # 5. Flip horizontally + brightness change
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_flip_left_right(
        img_tensor, seed=(1, 1))
    augmented = tf.image.stateless_random_brightness(
        augmented, max_delta=0.15, seed=(5, 6))
    data.append(augmented)
    
    # 6. Increase saturation significantly
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_saturation(
        img_tensor, lower=1.5, upper=2.0, seed=(6, 7))  # Much stronger saturation
    data.append(augmented)
    
    # 7. Decrease saturation (more grayscale-like)
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_saturation(
        img_tensor, lower=0.0, upper=0.5, seed=(7, 8))  # Much lower saturation
    data.append(augmented)
    
    # 8. Combined: contrast + saturation + brightness
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_contrast(
        img_tensor, lower=1.2, upper=1.5, seed=(8, 9))
    augmented = tf.image.stateless_random_saturation(
        augmented, lower=1.2, upper=1.5, seed=(9, 10))
    augmented = tf.image.stateless_random_brightness(
        augmented, max_delta=0.1, seed=(10, 11))
    data.append(augmented)
    
    # 9. Combined: lower contrast + lower saturation + JPEG quality
    img_tensor = tf.identity(original_img_tensor)
    augmented = tf.image.stateless_random_contrast(
        img_tensor, lower=0.5, upper=0.8, seed=(11, 12))
    augmented = tf.image.stateless_random_saturation(
        augmented, lower=0.4, upper=0.7, seed=(12, 13))
    augmented = tf.image.stateless_random_jpeg_quality(
        augmented, min_jpeg_quality=70, max_jpeg_quality=90, seed=(13, 14))  # Lower quality
            
    data.append(augmented)
    
    return data

def augment_directory(input_dir, output_dir=None):
    """
    Apply augmentations to all images in a directory
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images, if None saves to input_dir
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from the directory
    img_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f.lower())[1] in img_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for filename in tqdm(image_files, desc=f"Augmenting images in {input_dir}"):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        # Skip if image loading failed
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Generate augmented images
        augmented_images = data_aug(img)
        
        # Save the augmented images
        for image in augmented_images:
            # Use UUID for unique filenames
            new_filename = f"{uuid.uuid1()}.jpg"
            output_path = os.path.join(output_dir, new_filename)
            
            # Convert tensor to numpy array and save
            cv2.imwrite(output_path, image.numpy())
    
    print(f"Augmentation complete. Generated {len(image_files) * 9} new images.")
