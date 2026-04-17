import zipfile
import os

# Path to the zip archive
archive_path = 'archive (2).zip'
extract_path = 'ASL_Alphabet_Dataset'

# Extract the dataset only if not already done
if not os.path.exists(extract_path):
    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed.")
else:
    print("Dataset already extracted. Skipping extraction.")

# Locate the training folder inside the extracted archive.
# The zip structure is:
#   ASL_Alphabet_Dataset/
#     asl_alphabet_train/
#       asl_alphabet_train/
#         A/ B/ ... Z/ del/ nothing/ space/
train_folder = os.path.join(extract_path, 'asl_alphabet_train', 'asl_alphabet_train')

if not os.path.exists(train_folder):
    # Fallback: maybe the inner nesting is absent
    train_folder = os.path.join(extract_path, 'asl_alphabet_train')

if not os.path.exists(train_folder):
    raise FileNotFoundError(
        f"Training folder not found under '{extract_path}'. "
        "Please verify the archive structure."
    )

# List all class subdirectories
letter_classes = [
    d for d in os.listdir(train_folder)
    if os.path.isdir(os.path.join(train_folder, d))
]
print(f"Found {len(letter_classes)} classes: {sorted(letter_classes)}")

# Count total images in the full training set
total_images = 0
for cls in letter_classes:
    cls_path = os.path.join(train_folder, cls)
    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    total_images += len(images)
    print(f"  {cls}: {len(images)} images")

print(f"\nTotal training images: {total_images}")
print(f"Training folder ready at: {train_folder}")

# Export the training folder path so subsequent scripts can reuse it
with open('train_folder.txt', 'w') as f:
    f.write(train_folder)

print("Training folder path saved to train_folder.txt")
