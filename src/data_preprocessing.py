import os
import numpy as np
import cv2

def load_images_from_folder(folder, label, img_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    return images, labels

def load_dataset(base_path, img_size=(128, 128)):
    cats, y_cats = load_images_from_folder(os.path.join(base_path, 'cats'), 0, img_size)
    dogs, y_dogs = load_images_from_folder(os.path.join(base_path, 'dogs'), 1, img_size)
    X = np.array(cats + dogs)
    y = np.array(y_cats + y_dogs)
    return X, y

def preprocess_data(X):
    # Normalize pixel values
    return X.astype('float32') / 255.0