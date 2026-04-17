import pickle
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load splits to reconstruct the training generator's class indices
with open('splits.pkl', 'rb') as f:
    X_train_paths, _, _, y_train, _, _ = pickle.load(f)

train_df = pd.DataFrame({'filename': X_train_paths, 'class': y_train})
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical',
)

class_indices = train_generator.class_indices
# Invert: {index -> class_name}
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print(f"Saved {len(class_names)} class names to class_names.txt")
print("Classes:", class_names)
