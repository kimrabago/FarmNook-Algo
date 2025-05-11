from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# --- Firebase Configuration ---
PRICING_RULES_COLLECTION = "pricing_rules"
PRICING_RULES_DOC_ID = "ZTuDQiNR2KbFH0S2g6qV" # <--- IMPORTANT: Change this to your actual document ID in Firebase

# === NEW: Define your local Firebase credentials JSON filename ===
# === Make sure this file is in the SAME FOLDER as this Python script ===
FIREBASE_CREDENTIALS_LOCAL_FILENAME = "serviceAccountKey.json" # <--- !!! REPLACE THIS WITH YOUR ACTUAL FILENAME !!!

# Global variable to store pricing rules loaded from Firebase
FIREBASE_PRICING_RULES = {}
db = None

def initialize_firebase_and_load_rules():
    global FIREBASE_PRICING_RULES, db
    cred_object = None
    credentials_source_message = ""

    try:
        # 1. Try to use GOOGLE_APPLICATION_CREDENTIALS environment variable (best for production/cloud)
        env_cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if env_cred_path:
            if os.path.exists(env_cred_path):
                cred_object = credentials.Certificate(env_cred_path)
                credentials_source_message = f"using credentials from GOOGLE_APPLICATION_CREDENTIALS: {env_cred_path}"
            else:
                print(f"⚠️ Warning: GOOGLE_APPLICATION_CREDENTIALS is set but path '{env_cred_path}' does not exist.")
        
        # 2. If environment variable not used or failed, try the local file
        if not cred_object:
            # Construct the absolute path to the local credentials file relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_cred_path = os.path.join(script_dir, FIREBASE_CREDENTIALS_LOCAL_FILENAME)

            if os.path.exists(local_cred_path):
                cred_object = credentials.Certificate(local_cred_path)
                credentials_source_message = f"using local credentials file: {local_cred_path}"
            else:
                print(f"🔴 Error: Firebase credentials not found. \n"
                      f"   - GOOGLE_APPLICATION_CREDENTIALS environment variable not set or invalid.\n"
                      f"   - Local file '{FIREBASE_CREDENTIALS_LOCAL_FILENAME}' not found in script directory '{script_dir}'.")
                print("   Firebase pricing rules will not be loaded.")
                return

        # Check if Firebase app is already initialized to prevent re-initialization error
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred_object)
            print(f"✅ Firebase Admin SDK initialized successfully, {credentials_source_message}.")
        else:
            print(f"ℹ️ Firebase Admin SDK already initialized, {credentials_source_message}.")

        db = firestore.client()

        doc_ref = db.collection(PRICING_RULES_COLLECTION).document(PRICING_RULES_DOC_ID)
        doc = doc_ref.get()

        if doc.exists:
            FIREBASE_PRICING_RULES = doc.to_dict()
            if FIREBASE_PRICING_RULES:
                 print(f"✅ Pricing rules loaded successfully from Firebase: {PRICING_RULES_COLLECTION}/{PRICING_RULES_DOC_ID}")
            else:
                print(f"⚠️ Warning: Pricing rules document '{PRICING_RULES_DOC_ID}' in collection '{PRICING_RULES_COLLECTION}' is empty or not structured as expected.")
                FIREBASE_PRICING_RULES = {} # Ensure it's an empty dict
        else:
            print(f"🔴 Error: Pricing rules document '{PRICING_RULES_DOC_ID}' not found in Firebase collection '{PRICING_RULES_COLLECTION}'.")
            FIREBASE_PRICING_RULES = {} # Ensure it's an empty dict

    except Exception as e:
        print(f"🔴 Failed to initialize Firebase or load pricing rules: {e}")
        FIREBASE_PRICING_RULES = {} # Ensure it's an empty dict in case of any error

# Initialize Firebase and load rules when the application starts
initialize_firebase_and_load_rules()

# ... (rest of your Flask code remains the same as in the previous answer) ...

# Load model training dataset
try:
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

    model_pipeline.fit(X, y)
    print("✅ Model training complete.")
except FileNotFoundError:
    print("🔴 Error: 'final_datasets.csv' not found. Model training skipped.")
    model_pipeline = None # Ensure model_pipeline is None if training fails
except ValueError as e:
    print(f"🔴 Error during model training: {e}")
    model_pipeline = None # Ensure model_pipeline is None if training fails
except Exception as e:
    print(f"🔴 An unexpected error occurred during dataset loading or model training: {e}")
    model_pipeline = None


def estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance):
    rules = FIREBASE_PRICING_RULES.get(vehicle_type)
    if not rules:
        return -1 

    base = rules.get("base_fee", 0) 
    weight_cost_multiplier = rules.get("weight_fee", 0)
    pickup_fee_per_km = rules.get("pickup_fee", 0)
    delivery_fee_per_km = rules.get("delivery_fee", 0)
    base_km_included = rules.get("base_km", 0)

    weight_cost = weight_cost_multiplier * weight
    delivery_km_charged = max(delivery_distance - base_km_included, 0)
    pickup_cost = pickup_fee_per_km * pickup_distance
    delivery_cost_calculated = delivery_fee_per_km * delivery_km_charged

    return base + weight_cost + pickup_cost + delivery_cost_calculated

def get_recommendation_dt(input_data, pipeline_model, top_n=5):
    if not pipeline_model or not hasattr(pipeline_model.named_steps.get('classifier'), 'classes_'):
        return ["Model not trained yet or training failed."]

    try:
        probabilities = pipeline_model.predict_proba(input_data)
        vehicle_classes = pipeline_model.named_steps['classifier'].classes_
        prob_list = list(zip(vehicle_classes, probabilities[0]))
        prob_list.sort(key=lambda x: x[1], reverse=True)
        
        recommended_vehicles = [vehicle for vehicle, prob in prob_list if prob > 0][:top_n]
        
        if not recommended_vehicles:
             return ["No suitable vehicle found based on learned data or input is too dissimilar."]
        return recommended_vehicles
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error occurred during recommendation"]

@app.route("/recommend", methods=["POST"])
def recommend():
    if not model_pipeline:
        return jsonify({"error": "Model is not available due to training issues."}), 503

    try:
        data = request.json
        allowed_purposes = ["crops", "livestock", "perishable goods"]
        
        required_payload_keys = ["Product Type", "Product Weight (kg)", "Purpose"]
        for key in required_payload_keys:
            if key not in data:
                return jsonify({"error": f"Missing key in request: {key}"}), 400

        purpose = str(data.get("Purpose", "")).strip().lower()

        if purpose not in allowed_purposes:
            return jsonify({"error": f"Invalid purpose '{data.get('Purpose', '')}'. Must be one of: {', '.join(allowed_purposes)}"}), 400
        
        try:
            product_weight = float(data["Product Weight (kg)"])
            if product_weight <= 0:
                return jsonify({"error": "Product Weight (kg) must be greater than 0"}), 400
        except ValueError:
            return jsonify({"error": "Product Weight (kg) must be a valid number"}), 400

        input_df = pd.DataFrame([{
            "Product Type": str(data["Product Type"]),
            "Product Weight (kg)": product_weight,
            "Purpose": purpose
        }])
        recommendations = get_recommendation_dt(input_df, model_pipeline)
        return jsonify({"recommended_vehicle_types": recommendations})
    except KeyError as e: 
        return jsonify({"error": f"Missing data in request: {str(e)}"}), 400
    except Exception as e:
        print(f"🔥 Recommend Error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/estimate", methods=["POST"])
def estimate():
    if not FIREBASE_PRICING_RULES:
        return jsonify({"error": "Pricing rules not available. Please check Firebase configuration and connection."}), 503

    try:
        data = request.json
        
        required_estimate_keys = ["vehicleType", "weight", "pickupDistance", "deliveryDistance"]
        for key in required_estimate_keys:
            if key not in data:
                return jsonify({"error": f"Missing key in request: {key}"}), 400

        vehicle_type = data["vehicleType"]
        
        try:
            weight = float(data["weight"])
            business_to_pickup_km = float(data["pickupDistance"])
            pickup_to_dropoff_km = float(data["deliveryDistance"])
            
            if weight <= 0:
                return jsonify({"error": "Weight must be greater than 0"}), 400
            if business_to_pickup_km < 0 or pickup_to_dropoff_km < 0:
                return jsonify({"error": "Distances cannot be negative"}), 400

        except ValueError:
            return jsonify({"error": "Weight and distances must be valid numbers"}), 400

        estimated_cost = estimate_delivery_cost(vehicle_type, weight, business_to_pickup_km, pickup_to_dropoff_km)
        
        if estimated_cost == -1:
            return jsonify({"error": f"Unsupported vehicle type '{vehicle_type}' or rules not defined for it."}), 400

        return jsonify({
            "vehicleType": vehicle_type,
            "estimatedCost": round(estimated_cost, 2)
        })

    except KeyError as e: 
        return jsonify({"error": f"Missing data in request: {str(e)}"}), 400
    except Exception as e:
        print(f"🔥 Estimate Error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    print(f"🚀 Server starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)