import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    image_size = (128, 128)
    batch_size = 32

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important for correct evaluation!
    )

    print("Loading model...")
    model = tf.keras.models.load_model('cat_vs_dog_model.keras')

    print("Evaluating model...")
    loss, acc = model.evaluate(test_generator)
    print(f'Test accuracy: {acc:.4f}')

    # Predictions
    preds = (model.predict(test_generator) > 0.5).astype("int32")
    y_true = test_generator.classes

    print("Classification Report:")
    print(classification_report(y_true, preds, target_names=['Cat', 'Dog']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, preds))

if __name__ == "__main__":
    main()