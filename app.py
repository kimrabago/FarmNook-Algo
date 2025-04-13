from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("updated_dataset.csv")

# Define purpose mapping
def map_purpose(product_type):
    product = product_type.lower()
    if product in ["pigs", "cows", "goats", "chickens"]:
        return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]:
        return "goods"
    else:
        return "crops"

# Add purpose if not already there
if "Purpose" not in df.columns:
    df["Purpose"] = df["Product Type"].apply(map_purpose)

# Train classifier
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[["Product Type", "Purpose"]])
X_final = np.hstack((X_encoded, df[["Product Weight (kg)"]].values))
df = df.iloc[:len(X_final)].reset_index(drop=True)

model_category = DecisionTreeClassifier()
model_category.fit(X_final, df["Vehicle Category"])

STRICT_PURPOSES = ["livestock", "goods", "crops"]

# Flask API route
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    try:
        input_df = pd.DataFrame([{
            "Product Type": data["Product Type"],
            "Product Weight (kg)": float(data["Product Weight (kg)"]),
            "Purpose": data["Purpose"]
        }])
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400

    input_encoded = encoder.transform(input_df[["Product Type", "Purpose"]])
    input_final = np.hstack((input_encoded, np.array([[input_df["Product Weight (kg)"].values[0]]])))

    predicted_category = model_category.predict(input_final)[0]
    purpose = input_df["Purpose"].values[0]
    weight = input_df["Product Weight (kg)"].values[0]

    # Filter based on category and purpose
    filtered_df = df[df["Vehicle Category"] == predicted_category]

    if purpose in STRICT_PURPOSES:
        filtered_df = filtered_df[filtered_df["Purpose"] == purpose]

    if not filtered_df.empty:
        filtered_encoded = encoder.transform(filtered_df[["Product Type", "Purpose"]])
        filtered_final = np.hstack((filtered_encoded, filtered_df[["Product Weight (kg)"]].values))
        similarity_scores = cosine_similarity(input_final, filtered_final)
        best_idx = np.argmax(similarity_scores)
        closest_vehicle_type = filtered_df["Vehicle Type"].iloc[best_idx]
    else:
        closest_vehicle_type = "No suitable vehicle found"

    return jsonify({
        "vehicle_category": predicted_category,
        "vehicle_type": closest_vehicle_type
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)