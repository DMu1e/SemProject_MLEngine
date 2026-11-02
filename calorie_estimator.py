
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


# ==========================================
# 1. LOAD THE TRAINED FOOD IDENTIFICATION MODEL
# ==========================================

model = tf.keras.models.load_model("improved_food_cnn.keras")


# ==========================================
# 2. FOOD CATEGORIES
# ==========================================

class_names = [
    "bhaji", "chapati", "githeri", "kachumbari",
    "kukuchoma", "mandazi", "masalachips", "matoke",
    "mukimo", "nyamachoma", "pilau", "sukumawiki",
    "ugali"
]


# ==========================================
# 3. LOAD THE KENYAN FOOD DATABASE
# ==========================================

nutrition_data = pd.read_csv(
    "cleaned_kenya_food_composition_updated.csv"
)

# Remove unnecessary spaces from food names
nutrition_data["food_name"] = (
    nutrition_data["food_name"].astype(str).str.strip()
)



# ==========================================
# 4. MAP CNN LABELS TO DATABASE FOOD NAMES
# ==========================================

FOOD_DATABASE_MAPPING = {

    "bhaji": [
        "Potato Bhajia"
    ],

    "chapati": [
        "White Chapati"
    ],

    "githeri": [
        "Githeri (Fresh beans and Maize)"
    ],

    "kachumbari": [
    "Kachumbari"
],

"kukuchoma": [
    "Kuku Choma"
],


    "mandazi": [
        "Basic Mandazi (East African Doughnuts; Basic)"
    ],

    "masalachips": [],

    "matoke": [
        "Matoke (Stewed Green Bananas & Meat)"
    ],

    "mukimo": [
        "Mukimo (Fresh Maize, Potatoes & Pumpkin Leaves)"
    ],

    "nyamachoma": [
    "Nyama Choma"
     ],

    "pilau": [
        "Pilau (Spiced Rice)"
    ],

    "sukumawiki": [
        "Sukumawiki (Stir fried kales)"
    ],

    "ugali": [
        "Refined Maize Flour Ugali"
    ]

}

# ==========================================
# 5. IDENTIFY FOOD FOR CALORIE ESTIMATION
# ==========================================

# This is a separate function.
# Your original predict_food() remains unchanged.

def predict_food_for_calorie_lookup(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    predicted_food = class_names[predicted_index]

    print("\nFood Identification Results")
    print("---------------------------")

    print("Predicted food:", predicted_food)

    print(f"Confidence: {confidence:.2f}%")

    return predicted_food


# ==========================================
# 6. LOOK UP FOOD IN THE NUTRITION DATABASE
# ==========================================

def get_food_nutrition(food_name):

    database_names = FOOD_DATABASE_MAPPING.get(
        food_name, []
    )

    if not database_names:

        print("\nNutritional information is not")
        print("currently available for this food.")

        return None

    matching_foods = nutrition_data[
        nutrition_data["food_name"].isin(database_names)
    ]

    if matching_foods.empty:

        print("\nFood was not found in the database.")

        return None

    return matching_foods


# ==========================================
# 7. CALCULATE CALORIES
# ==========================================

def calculate_calories(calories_per_100g, weight):

    total_calories = (
        calories_per_100g * weight
    ) / 100

    return total_calories


# ==========================================
# 8. ESTIMATE FOOD CALORIES
# ==========================================

def estimate_food_calories(food_name):

    nutrition = get_food_nutrition(food_name)

    if nutrition is None:
        return

    print("\nAvailable Food Preparations")
    print("---------------------------")

    for index, row in nutrition.iterrows():

        print(
            f"{index + 1}. {row['food_name']} "
            f"({row['energy_kcal']} kcal per 100g)"
        )

    # Ask the user to select the preparation
    if len(nutrition) > 1:

        while True:

            try:

                choice = int(
                    input("\nSelect food preparation: ")
                )

                if 1 <= choice <= len(nutrition):
                    break

                print("Invalid selection. Try again.")

            except ValueError:

                print("Please enter a valid number.")

        selected_food = nutrition.iloc[choice - 1]

    else:

        selected_food = nutrition.iloc[0]

    # Get portion weight
    while True:

        try:

            weight = float(
                input("\nEnter food weight in grams: ")
            )

            if weight > 0:
                break

            print("Weight must be greater than zero.")

        except ValueError:

            print("Please enter a valid weight.")

    calories_per_100g = selected_food["energy_kcal"]

    total_calories = calculate_calories(
        calories_per_100g,
        weight
    )

    print("\nCalorie Estimation Results")
    print("--------------------------")

    print("Food:", selected_food["food_name"])

    print(f"Portion weight: {weight:.2f} g")

    print(
        f"Calories per 100g: "
        f"{calories_per_100g:.2f} kcal"
    )

    print(
        f"Estimated total calories: "
        f"{total_calories:.2f} kcal"
    )


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    image_path = input("Enter the path to your food image: ").strip().strip('"')

    try:

        predicted_food = predict_food_for_calorie_lookup(
            image_path
        )

        estimate_food_calories(predicted_food)

    except FileNotFoundError:

        print("\nError: Image, model or CSV file was not found.")

    except Exception as error:

        print("\nAn error occurred:", error)
