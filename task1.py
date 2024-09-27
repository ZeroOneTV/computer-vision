import os
import cv2

current_index = 0

def load_and_visualize_images(image_folder='images'):
    global current_index
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    while True:
        img_path = os.path.join(image_folder, image_files[current_index])
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imshow("Image Viewer", img)
            while True:
                key = cv2.waitKey(0)
                if key == 27:  # 'Esc'
                    cv2.destroyAllWindows()
                    return
                elif key == 81 or key == 0:  # Left arrow key or left button click
                    current_index = (current_index - 1) % len(image_files)
                    break
                elif key == 83 or key == 1:  # Right arrow key or right button click
                    current_index = (current_index + 1) % len(image_files)
                    break
        else:
            print(f"Failed to load image: {image_files[current_index]}")
            break
if __name__ == "__main__":
    load_and_visualize_images()
