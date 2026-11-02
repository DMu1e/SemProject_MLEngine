
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained CNN
model = tf.keras.models.load_model("improved_food_cnn.keras")

# Food categories
class_names = [
    "bhaji", "chapati", "githeri", "kachumbari",
    "kukuchoma", "mandazi", "masalachips", "matoke",
    "mukimo", "nyamachoma", "pilau", "sukumawiki", "ugali"
]

# Function to predict food from an image


def predict_food(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    print("Predicted food:", class_names[predicted_index])
    print(f"Confidence: {confidence:.2f}%")


# Test the model using your Mandazi image
predict_food(r"C:\Users\dalto\Downloads\Mukimo.jpg")
