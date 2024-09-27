import os
import cv2
import numpy as np

def process_images(input_folder='images', output_folder='processed_images', target_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, target_size)
            img_normalized = img_resized / 255.0
            img_array = np.array(img_normalized)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, (img_array * 255).astype(np.uint8))
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Failed to load image: {image_file}")

if __name__ == "__main__":
    process_images()