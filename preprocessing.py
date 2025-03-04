import tempfile
import subprocess
def convertToJPG(input_path):
    """
    Input: Original Image
    Output: JPEG image as bytes

    Converts an image to JPEG format using sips, returning the image data directly.
    """
    try:
        # Create a temporary file to store the JPEG
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp_file:
            # Convert to JPEG
            subprocess.run([
                'sips',
                '-s', 'format', 'jpeg',
                '--out', tmp_file.name,
                input_path
            ], check=True, capture_output=True)
            
            # Read the temporary file into memory
            with open(tmp_file.name, 'rb') as f:
                jpeg_data = f.read()
            
            return jpeg_data
            
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")
        return None

import cv2
import numpy as np
from mtcnn import MTCNN # Multitask Cascaded Convolutional Networks for face detection and alignment
from PIL import Image
import gc  # Garbage collector

def resize_large_image(image, max_size=1024):
    """Resize image if it's too large for MTCNN face detector"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size)
    return image

def detect_and_crop_face(jpeg_data, size=(250, 250), padding=0.4):
    """Detect and crop face with memory management"""
    try:
        # Initialize MTCNN detector
        detector = MTCNN()

        # Read and preprocess image
        nparr = np.frombuffer(jpeg_data, np.uint8)
        # Decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read image")
            return None
        
        # Preprocess large images
        image = resize_large_image(image)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        
        if not faces:
            print(f"No face detected")
            return None
        
        # Get largest face
        face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, width, height = face['box']
        
        # Calculate padding
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # Calculate coordinates
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + width + pad_x)
        y2 = min(image.shape[0], y + height + pad_y)
        
        # Extract and process face
        face_image = image[y1:y2, x1:x2]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = face_image.resize(size)
        
        # Convert to array and cleanup
        result = np.array(face_image)
        del face_image
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"Error processing: {str(e)}")
        return None


def process_img(image_path, output_path):
    jpeg_img = convertToJPG(image_path)
    cropped_face = detect_and_crop_face(jpeg_img)
    if cropped_face is not None:
        Image.fromarray(cropped_face).save(output_path)
        print(f"Processed: {image_path}")
    else:
        print(f"Failed to process: {image_path}")
