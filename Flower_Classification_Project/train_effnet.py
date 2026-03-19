import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tfrecords-jpeg-224x224')
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")

BATCH_SIZE = 32
IMAGE_SIZE = [224, 224]
NUM_CLASSES = 104

# === Stage 1: Train only the head (3 epochs) ===
STAGE1_EPOCHS = 3
# === Stage 2: Fine-tune the whole network (12 more epochs) ===
STAGE2_EPOCHS = 12

NUM_TRAINING_IMAGES = 12753
NUM_VALIDATION_IMAGES = 3712
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'class': tf.io.FixedLenFeature([], tf.int64),
}

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image

def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    return image

def read_tfrecord(example, augment=False):
    example = tf.io.parse_single_example(example, feature_description)
    image = decode_image(example['image'])
    if augment:
        image = data_augment(image)
    label = tf.cast(example['class'], tf.int32)
    return image, label

def load_dataset(filenames, augment=False):
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.with_options(opts)
    dataset = dataset.map(lambda x: read_tfrecord(x, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def get_training_dataset():
    filenames = tf.io.gfile.glob(TRAIN_DIR + "/*.tfrec")
    ds = load_dataset(filenames, augment=True).repeat().shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def get_validation_dataset():
    filenames = tf.io.gfile.glob(VAL_DIR + "/*.tfrec")
    ds = load_dataset(filenames).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def create_model():
    inputs = Input(shape=(*IMAGE_SIZE, 3))
    base = EfficientNetB4(weights='imagenet', include_top=False, input_tensor=inputs)
    base.trainable = False  # Stage 1: frozen backbone

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, base

def main():
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training directory not found at {TRAIN_DIR}")
        print("Please run download_dataset.py first.")
        return

    print("Loading datasets...")
    train_ds = get_training_dataset()
    val_ds   = get_validation_dataset()

    print("Building EfficientNetB4 model with ImageNet weights...")
    model, base = create_model()
    model.summary(line_length=80)

    # ===================== STAGE 1: Train the head only ========================
    print(f"\n{'='*60}")
    print("STAGE 1: Training classification head only...")
    print(f"{'='*60}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_s1 = ModelCheckpoint(
        "model_effnet_v2.h5",
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )
    
    history1 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=STAGE1_EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_s1]
    )
    print(f"Stage 1 done. Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

    # ===================== STAGE 2: Fine-tune whole network ====================
    print(f"\n{'='*60}")
    print("STAGE 2: Fine-tuning the full network...")
    print(f"{'='*60}")

    # Unfreeze backbone for fine-tuning with very small learning rate
    base.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_s2 = ModelCheckpoint(
        "model_effnet_v2.h5",
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )

    history2 = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=STAGE2_EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_s2, early_stop, reduce_lr]
    )
    
    best_acc = max(history2.history['val_accuracy'])
    print(f"\nTraining complete! Best val_accuracy: {best_acc:.4f}")
    print("Model saved to model_effnet_v2.h5")

if __name__ == "__main__":
    main()
