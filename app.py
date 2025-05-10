from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load model training dataset
df = pd.read_csv("final_datasets.csv")

# Ensure dataset includes required columns
required_cols = ["Product Type", "Purpose", "Product Weight (kg)", "Vehicle Type"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

#Data Cleaning
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
  "Motorcycle with Box":         { "base_fee": 80,   "weight_fee": 1.0,  "pickup_fee": 6.0,   "delivery_fee": 7.0, "base_km": 2 }, #1, 75
  "Tricycle":                    { "base_fee": 80,   "weight_fee": 0.8,  "pickup_fee": 6.0,   "delivery_fee": 9.0, "base_km": 2 },  #50-400
  "Small Multicab":              { "base_fee": 100,  "weight_fee": 1.0,  "pickup_fee": 7.0,   "delivery_fee": 12.0, "base_km": 3 },  # 400-1000, close to sedan
  "Large Multicab":              { "base_fee": 115,  "weight_fee": 0.9,  "pickup_fee": 8.0,   "delivery_fee": 14.0, "base_km": 3 },  # 900-1500, close to SUV
  "Small Delivery Van":          { "base_fee": 200,  "weight_fee": 0.8,  "pickup_fee": 9.0,   "delivery_fee": 16.0, "base_km": 5 },  # 500-1200, 600kg MPV
  "Large Delivery Van":          { "base_fee": 240,  "weight_fee": 0.7,  "pickup_fee": 10.0,  "delivery_fee": 18.0, "base_km": 5 }, # 1100-2000, 800kg pickup
  "Small Pickup Truck":          { "base_fee": 280,  "weight_fee": 0.7,  "pickup_fee": 10.0,  "delivery_fee": 17.0, "base_km": 5 }, # 700-1200, 1000kg van
  "Medium Pickup Truck":         { "base_fee": 940,  "weight_fee": 0.6,  "pickup_fee": 11.0,  "delivery_fee": 18.0, "base_km": 5 }, # 1100-1500,2000kg FB
  "Large Pickup Truck":          { "base_fee": 1040, "weight_fee": 0.5,  "pickup_fee": 12.0,  "delivery_fee": 20.0, "base_km": 5 }, # 1900-3500, 2000kg aluminum
  "Heavy Duty Pickup Truck":     { "base_fee": 1450, "weight_fee": 0.4,  "pickup_fee": 13.0,  "delivery_fee": 22.0, "base_km": 5 }, # 3400-4500
  "Dropside Truck":              { "base_fee": 4420, "weight_fee": 0.3,  "pickup_fee": 14.0,  "delivery_fee": 23.0, "base_km": 5 }, # 4400-9000
  "Elf Truck":                   { "base_fee": 1040, "weight_fee": 0.35, "pickup_fee": 13.0,  "delivery_fee": 22.0, "base_km": 5 }, # 1400-3500,close to 2000kg aluminum
  "10 Wheeler Cargo Truck":      { "base_fee": 7200, "weight_fee": 0.2,  "pickup_fee": 30.0,  "delivery_fee": 45.0, "base_km": 5 }, # 8000-12000
  "10 Wheeler Dump Truck":       { "base_fee": 7500, "weight_fee": 0.2,  "pickup_fee": 32.0,  "delivery_fee": 48.0, "base_km": 5 }, # 11000-15000

  # PERISHABLE GOODS
  "Small Refrigerated Van":      { "base_fee": 200,  "weight_fee": 0.8,  "pickup_fee": 9.0,   "delivery_fee": 18.0, "base_km": 5 }, # 1-800
  "Medium Refrigerated Van":     { "base_fee": 240,  "weight_fee": 0.7,  "pickup_fee": 10.0,  "delivery_fee": 19.0, "base_km": 5 }, # 801-1200
  "Large Refrigerated Van":      { "base_fee": 280,  "weight_fee": 0.6,  "pickup_fee": 11.0,  "delivery_fee": 20.0, "base_km": 5 }, # 1201-1500
  "Small Refrigerated Truck":    { "base_fee": 940,  "weight_fee": 0.5,  "pickup_fee": 11.0,  "delivery_fee": 21.0, "base_km": 5 }, # 1501-3000
  "Medium Refrigerated Truck":   { "base_fee": 1450, "weight_fee": 0.4,  "pickup_fee": 12.0,  "delivery_fee": 22.0, "base_km": 5 }, # 3001-5000
  "Large Refrigerated Truck":    { "base_fee": 4420, "weight_fee": 0.3,  "pickup_fee": 13.0,  "delivery_fee": 24.0, "base_km": 5 }, # 5001-9000
  "10 Wheeler Reefer Truck":     { "base_fee": 7200, "weight_fee": 0.2,  "pickup_fee": 35.0,  "delivery_fee": 50.0, "base_km": 5 }, # 9001-12000

  # LIVESTOCK
  "Small Livestock Truck":       { "base_fee": 240,  "weight_fee": 0.7,  "pickup_fee": 10.0,  "delivery_fee": 18.0, "base_km": 5 }, # 1-1500
  "Medium Livestock Truck":      { "base_fee": 1040, "weight_fee": 0.5,  "pickup_fee": 11.0,  "delivery_fee": 21.0, "base_km": 5 }, # 1501-3000
  "Large Livestock Truck":       { "base_fee": 1450, "weight_fee": 0.3,  "pickup_fee": 13.0,  "delivery_fee": 24.0, "base_km": 5 }, # 3001-7000
  "10 Wheeler Livestock Truck":  { "base_fee": 7200, "weight_fee": 0.2,  "pickup_fee": 33.0,  "delivery_fee": 46.0, "base_km": 5 }  # 7001-12000
}

def estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance):
    rules = pricing_rules.get(vehicle_type)
    if not rules:
        return -1

    base = rules["base_fee"]
    weight_cost = rules["weight_fee"] * weight

    # Delivery (with cargo) is from pickup to drop-off
    delivery_km_charged = max(delivery_distance - rules.get("base_km", 0), 0)

    pickup_cost = rules["pickup_fee"] * pickup_distance     # empty run
    delivery_cost = rules["delivery_fee"] * delivery_km_charged  # loaded run

    return base + weight_cost + pickup_cost + delivery_cost

def get_recommendation_dt(input_data, pipeline_model, top_n=5):
    if not hasattr(pipeline_model.named_steps['classifier'], 'classes_'):
        return ["Model not trained yet or training failed."]

    try:
        probabilities = pipeline_model.predict_proba(input_data)
        vehicle_classes = pipeline_model.named_steps['classifier'].classes_
        prob_list = list(zip(vehicle_classes, probabilities[0]))
        prob_list.sort(key=lambda x: x[1])
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
        allowed_purposes = ["crops", "livestock", "perishable goods"]
        purpose = data["Purpose"].strip().lower()

        # Validate purpose
        if purpose not in allowed_purposes:
            return jsonify({"error": f"Invalid purpose '{purpose}'. Must be one of: {', '.join(allowed_purposes)}"}), 400

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
        vehicle_type = data["vehicleType"]
        weight = float(data["weight"])
        business_to_pickup_km = float(data["pickupDistance"])
        pickup_to_dropoff_km = float(data["deliveryDistance"])

        estimated_cost = estimate_delivery_cost(vehicle_type, weight, business_to_pickup_km, pickup_to_dropoff_km)
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
