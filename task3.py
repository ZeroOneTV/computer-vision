import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.utils import image_dataset_from_directory
from keras.src.utils import load_img,img_to_array

base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

image_size = (224, 224)
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print("Class names:", class_names)

def preprocess_image(image):
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image)
    return image, label

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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.2f}')

def predict_image(image_path):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)
    class_label = class_names[class_idx[0]]
    return class_label

# Example
new_image_path = 'dataset/valid/0/5_1351088028_png.rf.23b179fbcc55c993c304bf283f80f607.jpg'
predicted_class = predict_image(new_image_path)
print(f'Predicted class: {predicted_class}')
