import cv2
import numpy as np

def letterbox_resize(image, height, width):
    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    
    # Calculate new dimensions while maintaining aspect ratio
    if aspect_ratio > width / height:
        new_width = width
        new_height = int(width / aspect_ratio)
    else:
        new_height = height
        new_width = int(height * aspect_ratio)
    
    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate coordinates to paste resized image in the center
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    
    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    return canvas