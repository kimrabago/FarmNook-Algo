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

# Pricing rules per vehicle type
pricing_rules = {
    "Tricycle": {"base_fee": 100, "weight_fee": 1.0, "pickup_fee": 5.0, "delivery_fee": 10.0},
    "Small Multicab": {"base_fee": 150, "weight_fee": 1.2, "pickup_fee": 6.0, "delivery_fee": 12.0},
    "Large Multicab": {"base_fee": 200, "weight_fee": 1.5, "pickup_fee": 7.0, "delivery_fee": 14.0},
    "Small Delivery Van": {"base_fee": 250, "weight_fee": 1.7, "pickup_fee": 8.0, "delivery_fee": 16.0},
    "Large Delivery Van": {"base_fee": 300, "weight_fee": 2.0, "pickup_fee": 9.0, "delivery_fee": 18.0},

    # 🚚 New entries
    "Motorcyle with Box": {"base_fee": 80, "weight_fee": 0.5, "pickup_fee": 3.0, "delivery_fee": 5.0},
    "Medium Pickup Truck": {"base_fee": 350, "weight_fee": 2.0, "pickup_fee": 10.0, "delivery_fee": 18.0},
    "Large Pickup Truck": {"base_fee": 400, "weight_fee": 2.5, "pickup_fee": 11.0, "delivery_fee": 20.0},
    "Heavy Duty Pickup Truck": {"base_fee": 500, "weight_fee": 3.0, "pickup_fee": 12.0, "delivery_fee": 22.0},
    "Dropside Truck": {"base_fee": 600, "weight_fee": 3.2, "pickup_fee": 14.0, "delivery_fee": 24.0},
    "Elf Truck": {"base_fee": 550, "weight_fee": 2.8, "pickup_fee": 13.0, "delivery_fee": 23.0},
    "10 Wheeler Cargo Truck": {"base_fee": 900, "weight_fee": 4.0, "pickup_fee": 20.0, "delivery_fee": 30.0},
    "10 Wheeler Dump Truck": {"base_fee": 950, "weight_fee": 4.2, "pickup_fee": 22.0, "delivery_fee": 32.0},

    "Small Refrigerated Van": {"base_fee": 300, "weight_fee": 2.2, "pickup_fee": 9.0, "delivery_fee": 18.0},
    "Medium Refrigerated Van": {"base_fee": 350, "weight_fee": 2.4, "pickup_fee": 10.0, "delivery_fee": 19.0},
    "Large Refrigerated Van": {"base_fee": 400, "weight_fee": 2.6, "pickup_fee": 11.0, "delivery_fee": 20.0},
    "Medium Refrigerated Truck": {"base_fee": 450, "weight_fee": 2.8, "pickup_fee": 12.0, "delivery_fee": 22.0},
    "Large Refrigerated Truck": {"base_fee": 500, "weight_fee": 3.0, "pickup_fee": 13.0, "delivery_fee": 24.0},
    "10 wheeler Reefer Truck": {"base_fee": 550, "weight_fee": 3.5, "pickup_fee": 15.0, "delivery_fee": 25.0},

    "Small Livestock Truck": {"base_fee": 350, "weight_fee": 2.0, "pickup_fee": 10.0, "delivery_fee": 18.0},
    "Medium Livestock Truck": {"base_fee": 450, "weight_fee": 2.5, "pickup_fee": 12.0, "delivery_fee": 22.0},
    "Large Livestock Truck": {"base_fee": 550, "weight_fee": 3.0, "pickup_fee": 14.0, "delivery_fee": 26.0},
    "10 wheeler Livestock Truck": {"base_fee": 700, "weight_fee": 3.5, "pickup_fee": 16.0, "delivery_fee": 28.0},
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

def estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance):
    rules = pricing_rules.get(vehicle_type)
    if not rules:
        return -1
    base = rules["base_fee"]
    weight_cost = rules["weight_fee"] * weight
    pickup_cost = rules["pickup_fee"] * pickup_distance
    delivery_cost = rules["delivery_fee"] * delivery_distance
    return base + weight_cost + pickup_cost + delivery_cost

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

@app.route("/estimate", methods=["POST"])
def estimate():
    try:
        data = request.json
         print("🔍 Received estimate request:", data)  # Add this
        vehicle_type = data["vehicleType"]
        weight = float(data["weight"])
        pickup_distance = float(data["pickupDistance"])
        delivery_distance = float(data["deliveryDistance"])

        estimated_cost = estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance)
        if estimated_cost == -1:
            return jsonify({"error": "Unsupported vehicle type"}), 400

        return jsonify({
            "vehicleType": vehicle_type,
            "estimatedCost": round(estimated_cost, 2)
        })

    except Exception as e:
        print("🔥 Estimate Error:", str(e))
        return jsonify({"error": str(e)}), 400
        
# === Run Flask App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)