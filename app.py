from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("farmnook_dataset_v2.csv")

# Purpose mapping
def map_purpose(product_type):
    product = product_type.lower()
    if product in ["pigs", "cows", "goats", "chickens"]:
        return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]:
        return "perishable goods"
    else:
        return "crops"

if "Purpose" not in df.columns:
    df["Purpose"] = df["Product Type"].apply(map_purpose)

# Vehicle type table by purpose
vehicle_table = {
    "crops": [
        ("Motorcyle with Box", 1, 200),
        ("Tricycle", 100, 500),
        ("Small Multicab", 400, 1000),
        ("Large Multicab", 900, 1500),
        ("Small Delivery Van", 500, 1200),
        ("Large Delivery Van", 1100, 2000),
        ("Small Pickup Truck", 700, 1200),
        ("Medium Pickup Truck", 1100, 1500),
        ("Large Pickup Truck", 1900, 3500),
        ("Heavy Duty Pickup Truck", 3400, 4500),
        ("Dropside Truck", 4400, 9000),
        ("Elf Truck", 1400, 3500),
        ("10 Wheeler Cargo Truck", 8000, 12000),
        ("10 Wheeler Dump Truck", 11000, 15000),
    ],
    "perishable goods": [
        ("Small Refrigerated Van", 1, 800),
        ("Medium Refrigerated Van", 801, 1200),
        ("Large Refrigerated Van", 1201, 1500),
        ("Small Refrigerated Truck", 1501, 3000),
        ("Medium Refrigerated Truck", 3001, 5000),
        ("Large Refrigerated Truck", 5001, 9000),
        ("10 wheeler Reefer Truck", 9001, 12000),
    ],
    "livestock": [
        ("Small Livestock Truck", 1, 1500),
        ("Medium Livestock Truck", 1501, 3000),
        ("Large Livestock Truck", 3001, 7000),
        ("10 wheeler Livestock Truck", 7001, 12000),
    ]
}

# Encode and prepare model data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[["Product Type", "Purpose"]])
X_final = np.hstack((X_encoded, df[["Product Weight (kg)"]].values))

min_len = min(len(X_final), len(df))
X_final = X_final[:min_len]
df = df.iloc[:min_len].reset_index(drop=True)

# Recommendation logic
def get_recommendation(input_data, encoder, product_column="Product Weight (kg)"):
    input_encoded = encoder.transform(input_data.drop(columns=[product_column]))
    input_final = np.hstack((input_encoded, np.array([[input_data[product_column].values[0]]])))

    purpose = input_data["Purpose"].values[0]
    weight = input_data[product_column].values[0]

    matched_vehicles = [
        vehicle for vehicle, min_wt, max_wt in vehicle_table.get(purpose, [])
        if min_wt <= weight <= max_wt
    ]

    filtered_df = df[
        (df["Purpose"] == purpose) &
        (df["Vehicle Type"].isin(matched_vehicles)) &
        (df["maxWeightCapacity"] >= weight)
    ]

    recommended_vehicles = []

    if not filtered_df.empty:
        filtered_encoded = encoder.transform(filtered_df[["Product Type", "Purpose"]])
        filtered_final = np.hstack((filtered_encoded, filtered_df[[product_column]].values))
        similarity_scores = cosine_similarity(input_final, filtered_final)

        sorted_indices = np.argsort(similarity_scores[0])[::-1]
        recommended_vehicles = filtered_df.iloc[sorted_indices]["Vehicle Type"].unique().tolist()
    else:
        recommended_vehicles = matched_vehicles

    if not recommended_vehicles:
        return ["No suitable vehicle found"]

    return recommended_vehicles

# === API Endpoint ===
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        input_df = pd.DataFrame([{
            "Product Type": data["Product Type"],
            "Product Weight (kg)": float(data["Product Weight (kg)"]),
            "Purpose": data["Purpose"]
        }])
        recommendations = get_recommendation(input_df, encoder)
        return jsonify({"recommended_vehicle_types": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# === Run Flask App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)