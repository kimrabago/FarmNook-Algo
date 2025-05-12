import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google.cloud import firestore
from google.oauth2 import service_account as google_service_account
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import traceback

# --- Load Environment Variables ---
load_dotenv()
print("ℹ️ Attempting to load environment variables...")

app = Flask(__name__)

db = None
try:
    print("ℹ️ Initializing Firestore using individual environment variables...")
    service_account_info = {
        "type": os.getenv("TYPE"),
        "project_id": os.getenv("PROJECT_ID"),
        "private_key_id": os.getenv("PRIVATE_KEY_ID"),
        "private_key": os.getenv("PRIVATE_KEY"),
        "client_email": os.getenv("CLIENT_EMAIL"),
        "client_id": os.getenv("CLIENT_ID"),
        "auth_uri": os.getenv("AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("UNIVERSE_DOMAIN", "googleapis.com")
    }

    required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id", "client_x509_cert_url"]
    missing_vars = [key for key in required_keys if not service_account_info.get(key)]

    if missing_vars:
        print(f"🔴🔴🔴 CRITICAL ERROR: Missing required environment variables for explicit credential setup: {missing_vars}")
        db = None
    elif not service_account_info["private_key"] or not service_account_info["private_key"].endswith("-----END PRIVATE KEY-----\n"):
         print(f"🔴🔴🔴 CRITICAL ERROR: The PRIVATE_KEY environment variable appears missing, incomplete, or incorrectly formatted.")
         db = None
    else:
        print("   Creating credentials object...")
        credentials = google_service_account.Credentials.from_service_account_info(service_account_info)
        db = firestore.Client(credentials=credentials, project=service_account_info["project_id"])
        print("✅ Firestore client initialized successfully using explicit environment variables.")

except Exception as e:
    print(f"🔴🔴🔴 CRITICAL ERROR: Failed to initialize Firestore client using explicit environment variables.")
    print(f"   Error details: {e}")
    print(f"   Potential causes: Missing/incorrect env vars, bad PRIVATE_KEY format, insufficient permissions, network issues.")
    print("--- Full Traceback ---")
    print(traceback.format_exc())
    print("--- End Traceback ---")
    db = None

# --- Configuration ---
FIRESTORE_RULES_COLLECTION = "pricing_rules"
FIRESTORE_RULES_DOCUMENT = "ZTuDQiNR2KbFH0S2g6qV"
LOADED_PRICING_RULES = {}
DATASET_FILE = "final_datasets.csv"

# --- Load Pricing Rules ---
def load_pricing_rules_from_firestore():
    """Fetches pricing rules from the specified Firestore document."""
    global LOADED_PRICING_RULES
    LOADED_PRICING_RULES = {}
    if not db:
        print("🔴 Firestore client not available. Cannot load pricing rules.")
        return
    try:
        print(f"ℹ️ Fetching pricing rules from Firestore: Collection='{FIRESTORE_RULES_COLLECTION}', Document='{FIRESTORE_RULES_DOCUMENT}'")
        doc_ref = db.collection(FIRESTORE_RULES_COLLECTION).document(FIRESTORE_RULES_DOCUMENT)
        doc_snapshot = doc_ref.get()
        if doc_snapshot.exists:
            rules_data = doc_snapshot.to_dict()
            if rules_data and isinstance(rules_data, dict):
                LOADED_PRICING_RULES = rules_data
                print(f"✅ Pricing rules loaded successfully. {len(LOADED_PRICING_RULES)} vehicle types found.")
            else:
                print(f"⚠️ Warning: Firestore document '{FIRESTORE_RULES_DOCUMENT}' exists but contains empty/invalid data.")
        else:
            print(f"🔴 Error: Firestore document '{FIRESTORE_RULES_DOCUMENT}' not found in collection '{FIRESTORE_RULES_COLLECTION}'.")
    except Exception as e:
        print(f"🔴 Error loading pricing rules from Firestore:")
        print(traceback.format_exc())

# --- Train ML Model ---
def train_recommendation_model():
    """Loads data and trains the vehicle recommendation model."""
    print("--- Loading Dataset and Training Model ---")
    try:
        print(f"ℹ️ Loading dataset from '{DATASET_FILE}'...")
        if not os.path.exists(DATASET_FILE):
            raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' not found.")

        df = pd.read_csv(DATASET_FILE)
        print(f"  Dataset loaded. Shape: {df.shape}")

        required_cols = ["Product Type", "Purpose", "Product Weight (kg)", "Vehicle Type"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset missing required columns: {required_cols}")

        df.dropna(subset=required_cols, inplace=True)
        df = df[df['Product Weight (kg)'] > 0]
        df = df[df['Vehicle Type'].notna() & (df['Vehicle Type'] != '')]

        if df.empty:
            raise ValueError("Dataset empty after cleaning. Cannot train model.")
        print(f"  Dataset shape after cleaning: {df.shape}")

        features = ["Product Type", "Purpose", "Product Weight (kg)"]
        target = "Vehicle Type"
        X = df[features]
        y = df[target]
        print(f"  Unique vehicle types in training data: {y.nunique()}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Product Type', 'Purpose']),
                ('num', 'passthrough', ['Product Weight (kg)'])],
            remainder='drop'
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5))])

        print("ℹ️ Starting model training...")
        pipeline.fit(X, y)
        print("✅ Model training complete.")
        print("--- Model Training Section Complete ---")
        return pipeline

    except FileNotFoundError as e:
        print(f"🔴 Error: {e}. Model training skipped.")
    except ValueError as e:
        print(f"🔴 Error during dataset processing or model training: {e}")
    except Exception as e:
        print(f"🔴 Unexpected error during model training:")
        print(traceback.format_exc())

    print("--- Model Training Section Complete (Failed) ---")
    return None

# --- Initialize Application Components ---
print("--- Loading Pricing Rules ---")
load_pricing_rules_from_firestore()
print("--- Pricing Rules Loading Complete ---")

model_pipeline = train_recommendation_model()


# --- Helper Function for Cost Estimation ---
def estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance):
    """Calculates delivery cost based on loaded rules."""
    if not LOADED_PRICING_RULES:
        print("⚠️ Cost estimation failed: Pricing rules not loaded.")
        return -3 if not db else -2 # Distinguish between client init failure and rules load failure

    rules = LOADED_PRICING_RULES.get(vehicle_type)
    if not rules:
        print(f"⚠️ Cost estimation failed: No rules found for vehicle type '{vehicle_type}'.")
        return -1 # Indicate vehicle type not found

    # Use .get with defaults for safety
    base = rules.get("base_fee", 0.0)
    weight_cost_multiplier = rules.get("weight_fee", 0.0)
    pickup_fee_per_km = rules.get("pickup_fee", 0.0)
    delivery_fee_per_km = rules.get("delivery_fee", 0.0)
    base_km_included = rules.get("base_km", 0.0)

    weight_cost = weight_cost_multiplier * weight
    delivery_km_charged = max(delivery_distance - base_km_included, 0)
    pickup_cost = pickup_fee_per_km * pickup_distance
    delivery_cost_calculated = delivery_fee_per_km * delivery_km_charged

    total_cost = base + weight_cost + pickup_cost + delivery_cost_calculated
    # Removed verbose internal logging here
    return total_cost

# --- Helper Function for Vehicle Recommendation ---
def get_recommendation_dt(input_data, pipeline_model, top_n=5):
    """Predicts vehicle types using the trained model pipeline."""
    if not pipeline_model:
        print("⚠️ Recommendation failed: Model pipeline unavailable.")
        return ["Model not available or training failed."]

    try:
        classifier = pipeline_model.named_steps.get('classifier')
        if not hasattr(classifier, 'classes_'):
             print("⚠️ Recommendation failed: Model classifier not fitted correctly.")
             return ["Model structure issue during recommendation."]

        # Removed input data logging
        probabilities = pipeline_model.predict_proba(input_data)
        vehicle_classes = classifier.classes_
        prob_list = sorted(zip(vehicle_classes, probabilities[0]), key=lambda x: x[1], reverse=True)

        recommended_vehicles = [
            {"type": vehicle, "probability": round(prob * 100, 2)}
            for vehicle, prob in prob_list if prob > 1e-6 # Filter out negligible probabilities
        ][:top_n]

        if not recommended_vehicles:
             print(f"⚠️ No suitable vehicle found with significant probability.")
             return ["No suitable vehicle type found based on prediction model."]

        print(f"✅ Recommendations generated: {[rec['type'] for rec in recommended_vehicles]}")
        return [rec['type'] for rec in recommended_vehicles]

    except Exception as e:
        print(f"🔴 Error during prediction:")
        print(traceback.format_exc())
        return ["Error occurred during recommendation process."]


# --- Flask API Endpoints ---

@app.route("/recommend", methods=["POST"])
def recommend():
    """API endpoint for recommending vehicle types."""
    print("\n--- Received /recommend request ---")
    if not model_pipeline:
        print("🔴 Responding 503: Recommendation model unavailable.")
        return jsonify({"error": "Recommendation model unavailable."}), 503

    try:
        data = request.json
        if not data:
            print("🔴 Responding 400: Request body not valid JSON.")
            return jsonify({"error": "Request body must be valid JSON."}), 400

        required_keys = ["Product Type", "Product Weight (kg)", "Purpose"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"🔴 Responding 400: Missing keys: {missing_keys}")
            return jsonify({"error": f"Missing key(s) in request: {', '.join(missing_keys)}"}), 400

        # Validate Purpose
        allowed_purposes = ["crops", "livestock", "perishable goods"]
        purpose = str(data.get("Purpose", "")).strip().lower()
        if purpose not in allowed_purposes:
            print(f"🔴 Responding 400: Invalid purpose '{data.get('Purpose', '')}'. Allowed: {allowed_purposes}")
            return jsonify({"error": f"Invalid purpose. Must be one of: {', '.join(allowed_purposes)}"}), 400

        # Validate Weight
        try:
            product_weight = float(data["Product Weight (kg)"])
            if product_weight <= 0:
                raise ValueError("Weight must be positive")
        except (ValueError, TypeError):
            print(f"🔴 Responding 400: Invalid Product Weight (kg): '{data.get('Product Weight (kg)')}'.")
            return jsonify({"error": "Product Weight (kg) must be a positive number."}), 400

        input_df = pd.DataFrame([{
            "Product Type": str(data["Product Type"]),
            "Product Weight (kg)": product_weight,
            "Purpose": purpose
        }])

        recommendations = get_recommendation_dt(input_df, model_pipeline)

        print(f"✅ Responding 200: Recommendations: {recommendations}")
        return jsonify({"recommended_vehicle_types": recommendations})

    except json.JSONDecodeError:
        print("🔴 Responding 400: Invalid JSON format.")
        return jsonify({"error": "Invalid JSON format in request body."}), 400
    except Exception as e:
        print(f"🔥 Recommend Endpoint Internal Error:")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route("/estimate", methods=["POST"])
def estimate():
    """API endpoint for estimating delivery cost."""
    print("\n--- Received /estimate request ---")

    if not db:
        print("🔴 Responding 503: Firestore client unavailable.")
        return jsonify({"error": "Service configuration error."}), 503
    if not LOADED_PRICING_RULES:
        print("🔴 Responding 503: Pricing rules unavailable.")
        return jsonify({"error": "Pricing rules unavailable."}), 503

    try:
        data = request.json
        if not data:
            print("🔴 Responding 400: Request body not valid JSON.")
            return jsonify({"error": "Request body must be valid JSON."}), 400

        required_keys = ["vehicleType", "weight", "pickupDistance", "deliveryDistance"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
             print(f"🔴 Responding 400: Missing keys: {missing_keys}")
             return jsonify({"error": f"Missing key(s) in request: {', '.join(missing_keys)}"}), 400

        vehicle_type = data.get("vehicleType")
        if not isinstance(vehicle_type, str) or not vehicle_type:
             print(f"🔴 Responding 400: Missing or invalid 'vehicleType': {vehicle_type}")
             return jsonify({"error": "Missing or invalid 'vehicleType' (must be a non-empty string)."}), 400

        # Validate numeric inputs
        try:
            weight = float(data["weight"])
            pickup_distance = float(data["pickupDistance"])
            delivery_distance = float(data["deliveryDistance"])
            if weight <= 0: raise ValueError("Weight must be positive")
            if pickup_distance < 0 or delivery_distance < 0: raise ValueError("Distances cannot be negative")
        except (ValueError, TypeError, KeyError) as e:
             print(f"🔴 Responding 400: Invalid numeric input for weight/distances. Data: {data}, Error: {e}")
             return jsonify({"error": "Invalid value for 'weight', 'pickupDistance', or 'deliveryDistance'. Must be valid, non-negative numbers (weight > 0)."}), 400

        estimated_cost = estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance)

        # Handle estimation results
        if estimated_cost == -3:
             print("🔴 Responding 503: Service configuration error (Firestore client).")
             return jsonify({"error": "Service configuration error."}), 503
        elif estimated_cost == -2:
             print("🔴 Responding 503: Pricing rules unavailable (load failed).")
             return jsonify({"error": "Pricing rules unavailable."}), 503
        elif estimated_cost == -1:
            print(f"🔴 Responding 404: Vehicle type '{vehicle_type}' not found in pricing rules.")
            return jsonify({"error": f"Pricing rules not found for vehicle type: '{vehicle_type}'."}), 404
        else:
            response_data = {
                "vehicleType": vehicle_type,
                "estimatedCost": round(estimated_cost, 2)
            }
            print(f"✅ Responding 200: Estimation successful: {response_data}")
            return jsonify(response_data)

    except json.JSONDecodeError:
        print("🔴 Responding 400: Invalid JSON format.")
        return jsonify({"error": "Invalid JSON format in request body."}), 400
    except Exception as e:
        print(f"🔥 Estimate Endpoint Internal Error:")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    is_model_ready = model_pipeline is not None
    are_rules_loaded = bool(LOADED_PRICING_RULES)
    firestore_ok = db is not None
    status = {
        "status": "ok",
        "auth_method": "explicit_env_vars",
        "firestore_client_initialized": firestore_ok,
        "model_ready": is_model_ready,
        "pricing_rules_loaded": are_rules_loaded and firestore_ok
    }
    # Overall health depends on core components being ready
    is_healthy = firestore_ok and is_model_ready and are_rules_loaded
    http_status = 200 if is_healthy else 503
    print(f"--- Responding to /health check: Status {http_status}, Healthy: {is_healthy} ---")
    return jsonify(status), http_status

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")

    print(f"--- Starting Flask Server ---")
    print(f"  Mode : {'DEBUG' if debug_mode else 'PRODUCTION'}")
    print(f"  Port : {port}")
    print(f"  Auth : Explicit Environment Variables (Not Recommended)")
    print(f"  Firestore Client Ready: {db is not None}")
    print(f"  Pricing Rules Loaded: {bool(LOADED_PRICING_RULES) and (db is not None)}")
    print(f"  Model Ready         : {model_pipeline is not None}")
    print(f"🚀 Server starting on http://0.0.0.0:{port}")

    # Use waitress or gunicorn for production instead of Flask's built-in server
    if not debug_mode:
        print(" K Running in production mode. Consider using Waitress or Gunicorn.")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)