from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("ph_vehicle_dataset_strict_corrected.csv")

# Add Purpose
def map_purpose(product_type):
    product = product_type.lower()
    if product in ["pigs", "cows", "goats", "chickens"]:
        return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]:
        return "perishable crops"
    else:
        return "crops"

if "Purpose" not in df.columns:
    df["Purpose"] = df["Product Type"].apply(map_purpose)

# Category Map
category_map = {
    "Multicab": "N1", "Tricycle": "L1", "Delivery Van": "N1",
    "Elf Truck": "N2", "Refrigerated Truck": "N2", "Livestock Truck": "N2",
    "Hilux": "N1", "Open Truck": "N1", "Refrigerated Van": "N1",
    "Mini Livestock Truck": "N1", "10-Wheeler Truck": "N3",
    "Heavy Livestock Trailer": "N3", "Container Truck": "N3"
}

if "Vehicle Category" not in df.columns:
    df["Vehicle Category"] = df["Vehicle Type"].map(category_map).fillna("N1")

# Encode
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[["Product Type", "Purpose"]])
X_final = np.hstack((X_encoded, df[["Product Weight (kg)"]].values))
df = df.iloc[:len(X_final)].reset_index(drop=True)

# Train model
model_category = DecisionTreeClassifier()
model_category.fit(X_final, df["Vehicle Category"])

STRICT_PURPOSES = ["livestock", "perishable crops", "crops"]

# Recommendation endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    input_df = pd.DataFrame([data])

    # Predict
    input_encoded = encoder.transform(input_df[["Product Type", "Purpose"]])
    input_final = np.hstack((input_encoded, np.array([[data["Product Weight (kg)"]]])))
    predicted_category = model_category.predict(input_final)[0]

    # Filter and score
    purpose = data["Purpose"]
    weight = data["Product Weight (kg)"]

    filtered_df = df[
        (df["Vehicle Category"] == predicted_category) &
        (df["maxWeightCapacity"] >= weight)
    ]
    if purpose in STRICT_PURPOSES:
        filtered_df = filtered_df[filtered_df["Purpose"] == purpose]

    if not filtered_df.empty:
        filtered_encoded = encoder.transform(filtered_df[["Product Type", "Purpose"]])
        filtered_final = np.hstack((filtered_encoded, filtered_df[["Product Weight (kg)"]].values))
        similarity_scores = cosine_similarity(input_final, filtered_final)
        best_idx = np.argmax(similarity_scores)
        vehicle_type = filtered_df["Vehicle Type"].iloc[best_idx]
    else:
        vehicle_type = "No suitable vehicle found"

    return jsonify({
        "vehicle_category": predicted_category,
        "vehicle_type": vehicle_type
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
