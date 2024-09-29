import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.utils import image_dataset_from_directory
from keras.src.utils import load_img,img_to_array
from keras.src.applications.resnet import ResNet50

base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

image_size = (224, 224)
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

def preprocess_image(image):
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image

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

base_model = ResNet50(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False
)

base_model.trainable = False

model = models.Sequential([
    base_model,  # Add the pre-trained ResNet50 base
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=15
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
