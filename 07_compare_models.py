import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Replace the placeholder values below with the actual test accuracies
# printed at the end of each training script.
# ---------------------------------------------------------------------------
accuracies = {
    'Landmark MLP':       0.86,   # from 03_landmark_model.py
    'MobileNetV2 Raw':    0.89,   # from 04_raw_cnn.py
    'EfficientNetB0 Raw': 0.03,   # from 04_raw_cnn.py
    'MobileNetV2 Crop':   0.91,   # from 06_crop_cnn.py
    'EfficientNetB0 Crop': 0.04,  # from 06_crop_cnn.py
}

models = list(accuracies.keys())
scores = list(accuracies.values())
colors = ['blue', 'green', 'green', 'orange', 'orange']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=colors)
plt.ylabel('Test Accuracy')
plt.title('Model Comparison — ASL Alphabet (Full Dataset)')
for bar, acc in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{acc:.3f}',
        ha='center',
    )
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

best_model_name = max(accuracies, key=accuracies.get)
print(f"Best model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")
print("Chart saved to model_comparison.png")
