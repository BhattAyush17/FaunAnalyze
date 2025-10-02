import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import build_model

def main():
    image_size = (128, 128)
    batch_size = 32

    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Validation data: only rescale
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    val_generator = val_datagen.flow_from_directory(
        'data/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = build_model(input_shape=(128, 128, 3))

    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,        # Stops after 3 epochs of no improvement
        restore_best_weights=True
    )

    model.fit(
        train_generator,
        epochs=30,  # You can use more epochs, early stopping will halt if needed
        validation_data=val_generator,
        callbacks=[early_stopping]
    )
    model.save('cat_vs_dog_model.keras')

if __name__ == "__main__":
    main()