from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load dataset
original_df = pd.read_csv("final_datasets.csv")

# Load model training dataset
df = pd.read_csv("final_datasets.csv")

# Define purpose mapping
def map_purpose(product_type):
    product = str(product_type).lower()
    if product in ["pigs", "cows", "goats", "chickens"]:
        return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]:
        return "perishable goods"
    elif product in ["corn", "rice", "fruit", "wheat", "vegetables"]:
        return "crops"
    else:
        return "crops"

if "Purpose" not in df.columns:
    if "Product Type" in df.columns:
        df["Purpose"] = df["Product Type"].apply(map_purpose)
    else:
        raise ValueError("Dataset must contain a 'Product Type' column.")

required_cols = ["Product Type", "Purpose", "Product Weight (kg)", "Vehicle Type"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

df.dropna(subset=required_cols, inplace=True)
df = df[df['Product Weight (kg)'] > 0]
df = df[df['Vehicle Type'].notna() & (df['Vehicle Type'] != '')]

if df.empty:
    raise ValueError("Dataset is empty after cleaning. Cannot train the model.")

features = ["Product Type", "Purpose", "Product Weight (kg)"]
target = "Vehicle Type"
X = df[features]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Product Type', 'Purpose']),
        ('num', 'passthrough', ['Product Weight (kg)'])
    ],
    remainder='drop'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5))
])

try:
    model_pipeline.fit(X, y)
    print("Model training complete.")
except ValueError as e:
    print(f"Error during model training: {e}")
    exit()

# Pricing rules per vehicle type
pricing_rules = {
    "Tricycle": {"base_fee": 100, "weight_fee": 1.0, "pickup_fee": 5.0, "delivery_fee": 10.0},
    "Small Multicab": {"base_fee": 150, "weight_fee": 1.2, "pickup_fee": 6.0, "delivery_fee": 12.0},
    "Large Multicab": {"base_fee": 200, "weight_fee": 1.5, "pickup_fee": 7.0, "delivery_fee": 14.0},
    "Small Delivery Van": {"base_fee": 250, "weight_fee": 1.7, "pickup_fee": 8.0, "delivery_fee": 16.0},
    "Large Delivery Van": {"base_fee": 300, "weight_fee": 2.0, "pickup_fee": 9.0, "delivery_fee": 18.0},
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

def estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance):
    rules = pricing_rules.get(vehicle_type)
    if not rules:
        return -1
    base = rules["base_fee"]
    weight_cost = rules["weight_fee"] * weight
    pickup_cost = rules["pickup_fee"] * pickup_distance
    delivery_cost = rules["delivery_fee"] * delivery_distance
    return base + weight_cost + pickup_cost + delivery_cost

def get_recommendation_dt(input_data, pipeline_model, top_n=5):
    if not hasattr(pipeline_model.named_steps['classifier'], 'classes_'):
        return ["Model not trained yet or training failed."]

    try:
        probabilities = pipeline_model.predict_proba(input_data)
        vehicle_classes = pipeline_model.named_steps['classifier'].classes_
        prob_list = list(zip(vehicle_classes, probabilities[0]))
        prob_list.sort(key=lambda x: x[1], reverse=True)
        recommended_vehicles = [vehicle for vehicle, prob in prob_list if prob > 0][:top_n]
        if not recommended_vehicles:
            return ["No suitable vehicle found based on learned data"]
        return recommended_vehicles
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error occurred during recommendation"]

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        input_df = pd.DataFrame([{
            "Product Type": data["Product Type"],
            "Product Weight (kg)": float(data["Product Weight (kg)"]),
            "Purpose": data["Purpose"]
        }])
        recommendations = get_recommendation_dt(input_df, model_pipeline)
        return jsonify({"recommended_vehicle_types": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/estimate", methods=["POST"])
def estimate():
    try:
        data = request.json
        print("🔍 Received estimate request:", data)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
