import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd

# ------------------------------
# Configuration
# ------------------------------
# Read the full training folder written by 01_extract_and_balance.py.
# Falls back to subset_5000 for quick experiments.
if os.path.exists('train_folder.txt'):
    with open('train_folder.txt', 'r') as _f:
        SOURCE_DIR = _f.read().strip()
else:
    SOURCE_DIR = 'subset_5000'

SKELETON_DIR = 'skeleton_images'   # output directory for skeleton images
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30                        # EarlyStopping will halt earlier if needed
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


# ------------------------------
# Step 1: Generate skeleton images
# ------------------------------
def generate_skeleton_images():
    """Create skeleton images from the source dataset and save to SKELETON_DIR."""
    os.makedirs(SKELETON_DIR, exist_ok=True)

    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    total = 0
    skipped = 0

    for cls in classes:
        src_class_path = os.path.join(SOURCE_DIR, cls)
        dst_class_path = os.path.join(SKELETON_DIR, cls)
        os.makedirs(dst_class_path, exist_ok=True)

        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(images, desc=f"Processing {cls}"):
            src_path = os.path.join(src_class_path, img_file)
            dst_path = os.path.join(dst_class_path, img_file)

            # Skip if already generated
            if os.path.exists(dst_path):
                total += 1
                continue

            image = cv2.imread(src_path)
            if image is None:
                print(f"Warning: cannot read {src_path}")
                skipped += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Draw skeleton on a white background
                skeleton = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

                for hand_landmarks in results.multi_hand_landmarks:
                    # Scale normalized landmarks to IMG_SIZE
                    points = [
                        (int(lm.x * IMG_SIZE), int(lm.y * IMG_SIZE))
                        for lm in hand_landmarks.landmark
                    ]

                    # Draw connections
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        cv2.line(skeleton, points[start_idx], points[end_idx], (0, 0, 0), 2)

                    # Draw landmark circles
                    for (x, y) in points:
                        cv2.circle(skeleton, (x, y), 3, (0, 0, 255), -1)

                cv2.imwrite(dst_path, skeleton)
                total += 1
            else:
                skipped += 1

    hands.close()
    print(f"Skeleton generation complete. Saved: {total}, Skipped (no hand): {skipped}")


# ------------------------------
# Step 2: Prepare data splits
# ------------------------------
def prepare_splits():
    """Gather all skeleton image paths and labels, split into train/val/test."""
    image_paths = []
    labels = []
    classes = [d for d in os.listdir(SKELETON_DIR) if os.path.isdir(os.path.join(SKELETON_DIR, d))]
    for cls in classes:
        class_path = os.path.join(SKELETON_DIR, cls)
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        image_paths.extend(images)
        labels.extend([cls] * len(images))

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------
# Step 3: Create model
# ------------------------------
def create_model(base_model_class, num_classes):
    base = base_model_class(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False  # Freeze pre-trained layers for fast training
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ------------------------------
# Step 4: Train and evaluate
# ------------------------------
def train_and_evaluate(model, train_gen, val_gen, test_gen, model_name, callbacks):
    print(f"\n--- Training {model_name} ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")
    return model, test_acc, history


# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    print(f"Source directory: {SOURCE_DIR}")

    # Step 1: generate skeletons (if not already done)
    skeleton_has_classes = (
        os.path.exists(SKELETON_DIR) and
        any(os.path.isdir(os.path.join(SKELETON_DIR, d)) for d in os.listdir(SKELETON_DIR))
    )
    if not skeleton_has_classes:
        print("Generating skeleton images...")
        generate_skeleton_images()
    else:
        print("Skeleton images already exist. Skipping generation.")

    # Step 2: prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits()

    with open('skeleton_splits.pkl', 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

    # Create dataframes for generators
    train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
    val_df   = pd.DataFrame({'filename': X_val,   'class': y_val})
    test_df  = pd.DataFrame({'filename': X_test,  'class': y_test})

    # Training augmentation for robustness; val/test rescale only
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest',
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
    )
    val_gen = eval_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
    )
    test_gen = eval_datagen.flow_from_dataframe(
        test_df, x_col='filename', y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False,
    )

    num_classes = len(train_gen.class_indices)

    # Callbacks: EarlyStopping + ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    ]

    # Step 4: Train MobileNetV2
    model_mnv2 = create_model(MobileNetV2, num_classes)
    model_mnv2, acc_mnv2, hist_mnv2 = train_and_evaluate(
        model_mnv2, train_gen, val_gen, test_gen, "MobileNetV2", callbacks
    )
    model_mnv2.save('mobilenetv2_skeleton.h5')

    # Train EfficientNetB0
    model_efnb0 = create_model(EfficientNetB0, num_classes)
    model_efnb0, acc_efnb0, hist_efnb0 = train_and_evaluate(
        model_efnb0, train_gen, val_gen, test_gen, "EfficientNetB0", callbacks
    )
    model_efnb0.save('efficientnetb0_skeleton.h5')

    # Step 5: Compare and save best model
    models_list = ['MobileNetV2', 'EfficientNetB0']
    accuracies = [acc_mnv2, acc_efnb0]
    best_idx = int(np.argmax(accuracies))
    best_model_name = models_list[best_idx]
    best_accuracy = accuracies[best_idx]
    best_model = model_mnv2 if best_idx == 0 else model_efnb0

    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")
    best_model.save('best_skeleton_model.h5')

    # Save class names
    class_names = [
        name for name, _ in sorted(train_gen.class_indices.items(), key=lambda item: item[1])
    ]
    with open('skeleton_class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Class names saved to skeleton_class_names.txt")

    # Plot comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models_list, accuracies, color=['blue', 'green'])
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison on Skeleton Images')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('skeleton_model_comparison.png')
    plt.show()

    print("All done. Best model saved as 'best_skeleton_model.h5'.")
