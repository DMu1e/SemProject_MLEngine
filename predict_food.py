
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import csv
import time

# 1. Load model
model = tf.keras.models.load_model(
    "multilabel_mobilenetv2_best_v1.keras",
    compile=False
)

# 2. Food labels
labels = [
    "bhaji", "chapati", "githeri", "kachumbari",
    "kukuchoma", "mandazi", "masalachips", "matoke",
    "mukimo", "nyamachoma", "pilau", "sukumawiki",
    "ugali", "samosa", "rice", "beef", "beef_stew",
    "beans", "avocado", "spaghetti", "chips", "tilapia"
]

# 3. Feedback storage
os.makedirs("feedback_images", exist_ok=True)

csv_file = "food_feedback.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"] + labels)

# 4. Select image
image_path = input("Enter food image path: ").strip().strip('"')

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

# 5. Predict
predictions = model.predict(image_array, verbose=0)[0]

print("\nMODEL PREDICTIONS")
print("=================")

for label, probability in sorted(
    zip(labels, predictions),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label}: {probability * 100:.2f}%")

# Display image
plt.imshow(image)
plt.axis("off")
plt.show()

# 6. Ask for correction
answer = input("\nWas the prediction correct? (y/n): ").strip().lower()

if answer == "n":

    print("\nEnter ALL foods present in the image.")
    print("Separate multiple foods with commas.")
    print("Available labels:", ", ".join(labels))

    actual_foods = [
        food.strip().lower()
        for food in input("\nActual foods: ").split(",")
        if food.strip()
    ]

    invalid = [food for food in actual_foods if food not in labels]

    if invalid:
        print("Invalid labels:", invalid)
        print("Correction not saved. Check spelling and try again.")

    elif not actual_foods:
        print("No labels entered. Correction not saved.")

    else:
        corrected_path = os.path.join(
            "feedback_images",
            f"feedback_{time.time_ns()}.jpg"
        )

        # Save original image at its available resolution
        original_image = Image.open(image_path).convert("RGB")
        original_image.save(corrected_path, "JPEG", quality=95)

        actual_labels = [
            1 if label in actual_foods else 0
            for label in labels
        ]

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([corrected_path] + actual_labels)

        print("\nCorrection saved successfully!")
        print("Actual foods:", actual_foods)

elif answer == "y":
    print("\nPrediction accepted.")

else:
    print("\nInvalid response. No correction saved.")

# 7. Show progress
with open(csv_file, "r", newline="") as f:
    total = sum(1 for row in csv.reader(f)) - 1

print(f"\nTotal corrected images collected: {total}")
print("No model training has taken place.")
