import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def create_augmentation_generator():
    """Create data augmentation generator"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
        fill_mode='nearest'
    )


class TrafficSignPreprocessor:
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size

    def load_image(self, image_path):
        """Load and preprocess a single image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        return img

    def preprocess_dataset(self, data_dir):
        """Preprocess entire dataset"""
        images = []
        labels = []

        for class_dir in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = self.load_image(img_path)
                    images.append(img)
                    labels.append(int(class_dir))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        return np.array(images), np.array(labels)