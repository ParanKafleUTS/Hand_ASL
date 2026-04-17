import cv2
import mediapipe as mp
import numpy as np
import os
import random
import argparse

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


def draw_landmarks_on_image(image_path, output_path=None):
    """
    Reads an image, detects hand landmarks, draws them, and either displays or saves the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        print(f"Hand detected in {image_path}")
    else:
        print(f"No hand detected in {image_path}")

    # Show or save
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to {output_path}")
    else:
        cv2.imshow("MediaPipe Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _get_train_dir():
    """Return the training directory from train_folder.txt if available, else fall back."""
    if os.path.exists('train_folder.txt'):
        with open('train_folder.txt', 'r') as f:
            return f.read().strip()
    # Legacy fallback for the 5k subset
    return 'subset_5000'


def pick_random_image(source_dir=None):
    """Pick a random image from source_dir (any class subfolder)."""
    if source_dir is None:
        source_dir = _get_train_dir()

    if not os.path.isdir(source_dir):
        print(f"Source directory not found: {source_dir}")
        return None

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    if not classes:
        print(f"No class folders found in {source_dir}.")
        return None

    random_class = random.choice(classes)
    class_path = os.path.join(source_dir, random_class)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No images found in class {random_class}")
        return None

    random_image = random.choice(images)
    return os.path.join(class_path, random_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw MediaPipe hand landmarks on an image.")
    parser.add_argument(
        "--image", type=str,
        help="Path to the image file. If not given, a random image from the training directory is used.",
    )
    parser.add_argument(
        "--source_dir", type=str, default=None,
        help="Directory containing class subfolders to pick a random image from. "
             "Defaults to path in train_folder.txt (written by 01_extract_and_balance.py).",
    )
    parser.add_argument(
        "--output", type=str,
        help="Optional output path to save the annotated image. "
             "If not given, the image is displayed on screen.",
    )
    args = parser.parse_args()

    if args.image:
        image_path = args.image
    else:
        image_path = pick_random_image(args.source_dir)
        if not image_path:
            print("No image available. Please provide an image path with --image.")
            exit(1)
        print(f"Using random image: {image_path}")

    draw_landmarks_on_image(image_path, args.output)
