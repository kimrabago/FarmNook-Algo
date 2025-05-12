import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google.cloud import firestore
from google.oauth2 import service_account as google_service_account
# --- Added for snapshot listener ---
import threading
from google.api_core.exceptions import Aborted
# --- End Added ---
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# --- Added for debugging ---
import traceback
import time # For potential retry logic or delays
# --- End Added ---

# --- Load Environment Variables ---
load_dotenv()
print("ℹ️ Attempting to load environment variables...")

app = Flask(__name__)

db = None
# --- Added for snapshot listener ---
# Global variable to hold the listener watcher object (optional, for potential cleanup)
rules_listener_watcher = None
# Event to signal when the first snapshot has been processed
rules_initial_load_complete = threading.Event()
# --- End Added ---

try:
    print("ℹ️ Initializing Firestore using individual environment variables...")
    private_key_raw = os.getenv("PRIVATE_KEY")
    if not private_key_raw:
        raise ValueError("PRIVATE_KEY environment variable is not set.")

    service_account_info = {
        "type": os.getenv("TYPE"),
        "project_id": os.getenv("PROJECT_ID"),
        "private_key_id": os.getenv("PRIVATE_KEY_ID"),
        # Replace literal '\n' strings with actual newline characters
        "private_key": private_key_raw.replace('\\n', '\n'),
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
         print(f"🔴🔴🔴 CRITICAL ERROR: The PRIVATE_KEY environment variable appears missing, incomplete, or incorrectly formatted (after potential replacements). Check the raw value.")
         print(f"   Starts with: {service_account_info['private_key'][:30]}...") # Log start
         print(f"   Ends with: ...{service_account_info['private_key'][-30:]}") # Log end
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
FIRESTORE_RULES_DOCUMENT = "ZTuDQiNR2KbFH0S2g6qV" # <-- Make absolutely sure this ID is correct!
LOADED_PRICING_RULES = {} # This will be updated by the listener
DATASET_FILE = "final_datasets.csv"


# --- Pricing Rules Snapshot Callback ---
def on_rules_snapshot(doc_snapshot, changes, read_time):
    """Callback function for the Firestore listener."""
    global LOADED_PRICING_RULES
    # --- DEBUG: Log entry into the callback ---
    print(f"\n✅✅✅ [Callback] on_rules_snapshot function ENTERED at {read_time} ✅✅✅")

    try:
        # --- DEBUG: Log snapshot details ---
        print(f"   [Callback] Received {len(doc_snapshot)} document(s) in snapshot.")
        # Check if snapshot list is not empty (it should contain one doc for a document listener)
        if not doc_snapshot:
            print("   [Callback] Snapshot list is empty. This is unusual for a single doc listener after initial connection.")
            # Potentially signal completion if this is unexpected on first load
            if not rules_initial_load_complete.is_set():
                 print("   [Callback] Signaling initial load complete (due to empty snapshot).")
                 rules_initial_load_complete.set()
            print(f"🏁🏁🏁 [Callback] on_rules_snapshot function EXITING (empty snapshot) 🏁🏁🏁")
            return # Exit if no documents

        single_doc = doc_snapshot[0] # Process the first document
        # --- DEBUG: Log document details ---
        print(f"   [Callback] Processing document ID: '{single_doc.id}', Path: '{single_doc.reference.path}', Exists: {single_doc.exists}")

        if single_doc.exists:
            print(f"   [Callback] Document exists. Attempting to convert to dict...")
            rules_data = single_doc.to_dict()
            # --- DEBUG: Log the data received from Firestore ---
            print(f"   [Callback] Raw Data from Firestore: {rules_data}")
            if rules_data and isinstance(rules_data, dict):
                print(f"   [Callback] Data is valid dict. Updating global LOADED_PRICING_RULES...")
                LOADED_PRICING_RULES = rules_data # Update the global variable
                print(f"✅ [Callback] Pricing rules updated successfully via snapshot. {len(LOADED_PRICING_RULES)} vehicle types loaded.")
                # --- DEBUG: Log the state of the global variable AFTER update ---
                print(f"   [Callback] Current LOADED_PRICING_RULES: {LOADED_PRICING_RULES}")
            else:
                print(f"⚠️ [Callback] Warning: Firestore document '{FIRESTORE_RULES_DOCUMENT}' updated but contains empty/invalid data (not a non-empty dict). Clearing rules.")
                LOADED_PRICING_RULES = {} # Clear rules if data is invalid
        else:
            # Handle case where the document is deleted
            print(f"🔴 [Callback] Warning: Firestore document '{FIRESTORE_RULES_DOCUMENT}' deleted or does not exist. Clearing loaded rules.")
            LOADED_PRICING_RULES = {}

        # Signal that the initial load (or an update) has happened
        if not rules_initial_load_complete.is_set():
             print("✅ [Callback] Initial pricing rules load from snapshot complete. Signaling event.")
             rules_initial_load_complete.set()
        else:
             print("ℹ️ [Callback] Subsequent snapshot update processed.")

    except Exception as e:
        # --- DEBUG: Log any error during callback processing ---
        print(f"🔴🔴🔴 [Callback] CRITICAL ERROR processing Firestore snapshot for pricing rules:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Args: {e.args}")
        print(traceback.format_exc())
        # Signal completion even on error to avoid blocking startup forever
        if not rules_initial_load_complete.is_set():
             print("⚠️ [Callback] Initial pricing rules load from snapshot failed. Signaling event anyway.")
             rules_initial_load_complete.set()

    # --- DEBUG: Log exit from the callback ---
    print(f"🏁🏁🏁 [Callback] on_rules_snapshot function EXITING 🏁🏁🏁\n")


# --- Function to Start the Listener ---
def start_rules_listener():
    """Initializes and starts the Firestore listener for pricing rules."""
    global rules_listener_watcher
    if not db:
        print("🔴 [Listener] Firestore client not available. Cannot start pricing rules listener.")
        rules_initial_load_complete.set() # Allow app to start, but rules won't load
        return

    try:
        # --- DEBUG: Log listener startup attempt ---
        print(f"ℹ️ [Listener] Attempting to start Firestore listener for: Collection='{FIRESTORE_RULES_COLLECTION}', Document='{FIRESTORE_RULES_DOCUMENT}'")
        doc_ref = db.collection(FIRESTORE_RULES_COLLECTION).document(FIRESTORE_RULES_DOCUMENT)
        # --- DEBUG: Log the exact path being watched ---
        print(f"   [Listener] Document reference created: '{doc_ref.path}'")

        # --- DEBUG: Log right before attaching ---
        print("   [Listener] Attaching snapshot listener...")
        # Watch the document for changes
        # Pass the callback function to on_snapshot
        rules_listener_watcher = doc_ref.on_snapshot(on_rules_snapshot)
        # --- DEBUG: Log success IF attaching didn't raise an immediate error ---
        # Note: Actual connection confirmation happens in the first callback run.
        print("✅ [Listener] Firestore listener attached successfully (or attachment call succeeded). Waiting for first snapshot event...")

    except Exception as e:
        # --- DEBUG: Log specific error during listener startup ---
        print(f"🔴🔴🔴 [Listener] CRITICAL ERROR starting Firestore listener for pricing rules:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Args: {e.args}")
        print(traceback.format_exc())
        # Ensure app doesn't hang waiting for initial load if listener fails
        if not rules_initial_load_complete.is_set():
             print("   [Listener] Signaling initial load 'complete' (due to listener start error) to prevent startup hang.")
             rules_initial_load_complete.set()

# --- Train ML Model (No debugging changes needed here, assuming it works) ---
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
        return None # Make sure to return None on error
    except ValueError as e:
        print(f"🔴 Error during dataset processing or model training: {e}")
        return None # Make sure to return None on error
    except Exception as e:
        print(f"🔴 Unexpected error during model training:")
        print(traceback.format_exc())
        return None # Make sure to return None on error

    print("--- Model Training Section Complete (Failed) ---")
    return None


# --- Initialize Application Components ---

# 1. Start the Firestore listener (it will run in the background)
print("--- Initializing Real-time Pricing Rules Listener ---")
start_rules_listener() # Contains detailed startup logs

# 2. Train the model (can happen concurrently or after listener starts)
model_pipeline = train_recommendation_model()

# 3. Wait for the *initial* rules load before proceeding (optional but recommended)
print("--- Waiting for initial pricing rules load from Firestore snapshot... (Max 30 seconds) ---")
# Wait for a reasonable time (e.g., 30 seconds) for the first snapshot
initial_load_ok = rules_initial_load_complete.wait(timeout=30.0)
if initial_load_ok:
    print("--- Initial pricing rules load signal received. ---")
else:
    print("⚠️ Warning: Timed out waiting for initial pricing rules snapshot signal. Rules might be empty or stale if callback hasn't run yet or failed.")
print("--- Initialization Potentially Complete (Proceeding to start server) ---")


# --- Helper Function for Cost Estimation (No changes needed here) ---
# Uses the globally updated LOADED_PRICING_RULES
def estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance):
    """Calculates delivery cost based on loaded rules."""
    # Access the global variable directly, which is updated by the listener
    current_rules = LOADED_PRICING_RULES # Reads the potentially updated global dict

    if not current_rules: # Check if the dictionary is empty
        print("⚠️ Cost estimation failed: Pricing rules dictionary is currently empty.")
         # Distinguish reasons: If db failed init vs. listener hasn't loaded/doc empty
        return -3 if not db else -2

    rules = current_rules.get(vehicle_type)
    if not rules:
        print(f"⚠️ Cost estimation failed: No rules found for vehicle type '{vehicle_type}' in currently loaded rules.")
        return -1 # Indicate vehicle type not found in current rules

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
    return total_cost


# --- Helper Function for Vehicle Recommendation (No changes needed) ---
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

        probabilities = pipeline_model.predict_proba(input_data)
        vehicle_classes = classifier.classes_
        prob_list = sorted(zip(vehicle_classes, probabilities[0]), key=lambda x: x[1], reverse=True)

        recommended_vehicles = [
            {"type": vehicle, "probability": round(prob * 100, 2)}
            for vehicle, prob in prob_list if prob > 1e-6
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
    # --- DEBUG: Log the state of rules WHEN the request is handled ---
    print(f"   [Estimate Endpoint] Current state of LOADED_PRICING_RULES at request time: {LOADED_PRICING_RULES}")
    # --- End Debug Log ---

    # Check if rules are available (they might be empty if doc deleted/invalid/listener failed)
    if not LOADED_PRICING_RULES and db: # Check specifically if rules are empty but db is ok
         print("🔴 Responding 503: Pricing rules unavailable (empty dictionary). Check listener logs.")
         return jsonify({"error": "Pricing rules unavailable or document deleted."}), 503
    elif not db: # Check if DB itself failed
        print("🔴 Responding 503: Firestore client unavailable.")
        return jsonify({"error": "Service configuration error (DB)."}), 503

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

        # Call the helper, which now uses the potentially updated global rules
        estimated_cost = estimate_delivery_cost(vehicle_type, weight, pickup_distance, delivery_distance)

        # Handle estimation results based on return codes from estimate_delivery_cost
        if estimated_cost == -3: # DB init failure
             print("🔴 Responding 503: Service configuration error (Firestore client).")
             return jsonify({"error": "Service configuration error (DB)."}), 503
        elif estimated_cost == -2: # Rules empty/not loaded, but DB is ok
             print("🔴 Responding 503: Pricing rules unavailable (empty dictionary). Check listener logs.")
             return jsonify({"error": "Pricing rules unavailable or document deleted."}), 503
        elif estimated_cost == -1: # Vehicle type not found in current rules
            print(f"🔴 Responding 404: Vehicle type '{vehicle_type}' not found in current pricing rules.")
            return jsonify({"error": f"Pricing rules not found for vehicle type: '{vehicle_type}'."}), 404
        else: # Success
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
    # Check if rules dictionary is populated *and* the initial load signal was set
    are_rules_loaded = bool(LOADED_PRICING_RULES) and rules_initial_load_complete.is_set()
    firestore_ok = db is not None
    status = {
        "status": "ok",
        "auth_method": "explicit_env_vars",
        "firestore_client_initialized": firestore_ok,
        "model_ready": is_model_ready,
        "pricing_rules_loaded": are_rules_loaded, # Reflects if dict has content AND initial load signaled
        "initial_rules_snapshot_received": rules_initial_load_complete.is_set(),
        "current_rules_snapshot_length": len(LOADED_PRICING_RULES) # Add length for quick check
    }
    # Overall health depends on core components being ready
    is_healthy = firestore_ok and is_model_ready and are_rules_loaded
    http_status = 200 if is_healthy else 503
    print(f"--- Responding to /health check: Status {http_status}, Healthy: {is_healthy}, Rules Loaded: {are_rules_loaded} ---")
    return jsonify(status), http_status

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")

    print(f"--- Starting Flask Server ---")
    print(f"  Mode : {'DEBUG' if debug_mode else 'PRODUCTION'}")
    print(f"  Port : {port}")
    print(f"  Auth : Explicit Environment Variables")
    print(f"  Firestore Client Ready: {db is not None}")
    # Note: Rules might not be loaded *at this exact moment* due to async listener start + wait timeout
    print(f"  Initial Rules Snapshot Received Signal: {rules_initial_load_complete.is_set()}")
    print(f"  Pricing Rules Initially Loaded (at server start time): {bool(LOADED_PRICING_RULES)}")
    print(f"  Model Ready         : {model_pipeline is not None}")
    print(f"🚀 Server starting on http://0.0.0.0:{port}")

    # Use waitress or gunicorn for production instead of Flask's built-in server
    if not debug_mode:
        print(" K Running in production mode. Using Waitress.")
        try:
            from waitress import serve
            serve(app, host="0.0.0.0", port=port)
        except ImportError:
            print("⚠️ Waitress not installed. Falling back to Flask development server (NOT RECOMMENDED for production).")
            print("   Install using: pip install waitress")
            app.run(host="0.0.0.0", port=port, debug=False) # Never run debug=True with dev server in prod
    else:
         # Use Flask's development server for debugging
        print(" K Running in DEBUG mode with Flask development server.")
        app.run(host="0.0.0.0", port=port, debug=True)