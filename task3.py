import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.src.utils import image_dataset_from_directory
from keras.src.utils import load_img,img_to_array

# Set paths for dataset directories
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Parameters
image_size = (224, 224)  # Desired image size
batch_size = 32  # Batch size for training
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = sorted(os.listdir(train_dir))  # This gets the folder names (e.g., '0', '1')
num_classes = len(class_names)
print("Class names:", class_names)

# 1. Function to load and preprocess the images
def preprocess_image(image):
    image = tf.image.resize(image, image_size)  # Resize to target size
    image = image / 255.0  # Normalize pixel values
    return image

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # or decode_png if you are using png images
    image = preprocess_image(image)
    return image, label

# 2. Function to create dataset
def create_dataset(directory):
    dataset = image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    return dataset.prefetch(buffer_size=AUTOTUNE)

train_dataset = create_dataset(train_dir)
valid_dataset = create_dataset(valid_dir)
test_dataset = create_dataset(test_dir)

# 3. Design the CNN model using Keras
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Number of classes
])

# 4. Compile the model using TensorFlow
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10  # Adjust the number of epochs as needed
)

# 6. Evaluate the model using test data
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.2f}')

# 7. Predict a new image
def predict_image(image_path):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)  # Get the index of the class with the highest probability
    class_label = class_names[class_idx[0]]  # Use class names from folder structure
    return class_label

# Example usage for predicting a new image
# new_image_path = 'path_to_your_image.jpg'
# predicted_class = predict_image(new_image_path)
# print(f'Predicted class: {predicted_class}')
