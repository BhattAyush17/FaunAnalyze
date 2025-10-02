import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(num_images=5):
    image_size = (128, 128)
    batch_size = num_images  # load only as many images as needed

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    model = tf.keras.models.load_model('cat_vs_dog_model.keras')
    X, y_true = next(test_generator)
    preds = (model.predict(X) > 0.5).astype("int32")

    for i in range(num_images):
        plt.imshow(X[i])
        plt.title(f'Predicted: {"Dog" if preds[i][0] else "Cat"}, Actual: {"Dog" if y_true[i] else "Cat"}')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    visualize_predictions()