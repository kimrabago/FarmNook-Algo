import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings # To suppress SettingWithCopyWarning if needed

# Suppress SettingWithCopyWarning (optional, use with caution)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


from google.colab import drive
drive.mount('/content/drive')

# Load the CSV
try:
    df = pd.read_csv("/content/drive/MyDrive/FarmNook/ph_vehicle_dataset_final_cleaned.csv")
    print("CSV loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please check the path.")
    # You might want to stop execution here or use a default DataFrame
    # For demonstration, creating a dummy DataFrame if file not found
    data = {'Product Type': ['Pigs', 'Fruits', 'Frozen Fish', 'Cows', 'Milk', 'Eggs', 'Leafy Greens'],
            'Product Weight (kg)': [1500, 500, 800, 2500, 600, 200, 150],
            'Vehicle Type': ['Livestock Truck', 'Multicab', 'Refrigerated Van', 'Heavy Livestock Trailer', 'Refrigerated Van', 'Tricycle', 'Tricycle'],
            'maxWeightCapacity': [2000, 600, 1000, 5000, 1000, 300, 300]}
    df = pd.DataFrame(data)
    print("Using dummy data for demonstration.")


# --- Feature Engineering ---
def map_purpose(product_type):
    product = product_type.lower()
    if product in ["pigs", "cows", "goats", "chickens"]: return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]: return "perishable crops"
    else: return "crops" # Default to non-perishable crops

# Apply map_purpose safely
if "Product Type" in df.columns:
    if "Purpose" not in df.columns:
        df["Purpose"] = df["Product Type"].apply(map_purpose)
else:
    print("Warning: 'Product Type' column not found. Cannot create 'Purpose'.")
    # Handle this case, maybe create a default 'Purpose' or stop


category_map = {
    "Multicab": "N1", "Tricycle": "L1", "Delivery Van": "N1",
    "Elf Truck": "N2", "Refrigerated Truck": "N2", "Livestock Truck": "N2",
    "Pickup Truck": "N1", "Open Truck": "N1", "Refrigerated Van": "N1",
    "Mini Livestock Truck": "N1", "10-Wheeler Truck": "N3",
    "Heavy Livestock Trailer": "N3", "Container Truck": "N3"
}
# Apply category_map safely
if "Vehicle Type" in df.columns:
     if "Vehicle Category" not in df.columns:
        df["Vehicle Category"] = df["Vehicle Type"].map(category_map).fillna("N1") # Fill unknowns as N1
else:
    print("Warning: 'Vehicle Type' column not found. Cannot create 'Vehicle Category'.")
    # Handle this case

# --- Check for required columns before proceeding ---
required_cols = ["Product Type", "Purpose", "Product Weight (kg)", "Vehicle Category", "Vehicle Type", "maxWeightCapacity"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns: {missing_cols}. Please check the CSV or data generation.")
    # Stop execution or handle appropriately
    exit() # Or raise an error

# --- One-Hot Encode ---
# Define features to encode BEFORE splitting
features_to_encode = ["Product Type", "Purpose"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit the encoder ONLY on the full dataset initially to learn all categories
encoder.fit(df[features_to_encode])

# Transform the full dataset (will be used later for splitting)
X_encoded = encoder.transform(df[features_to_encode])
X_final = np.hstack((X_encoded, df[["Product Weight (kg)"]].values))

# Define target variables BEFORE splitting
y_category = df["Vehicle Category"]
y_vehicle_type = df["Vehicle Type"] # For final recommendation evaluation

# --- Split Data into Training and Testing Sets ---
try:
    X_train, X_test, df_train, df_test = train_test_split(
        X_final, df, test_size=0.25, random_state=42, stratify=y_category
    )

    # Extract corresponding target variables for train/test sets
    y_category_train = df_train["Vehicle Category"]
    y_category_test = df_test["Vehicle Category"]
    y_vehicle_type_test = df_test["Vehicle Type"] # Actual vehicle types for test set

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

except ValueError as e:
    print(f"Error during train_test_split: {e}")
    print("This might happen if the dataset is too small or classes have very few members.")
    # Handle this - maybe skip evaluation or use cross-validation
    exit()


# --- Train Model ONLY on Training Data ---
model_category = DecisionTreeClassifier(random_state=42) # Added random_state for reproducibility
model_category.fit(X_train, y_category_train)

# --- Modify get_recommendation to use a specific candidate pool and threshold ---
STRICT_PURPOSES = ["livestock", "perishable crops", "crops"] # Redefined here for clarity

def get_recommendation_eval_with_threshold(
    input_data,
    encoder,
    model_category,
    df_candidates,
    product_weight_column="Product Weight (kg)",
    similarity_threshold=0.75 # <<< Added Threshold Parameter (default 0.75)
    ):
    """
    Modified recommendation function for evaluation with similarity threshold.
    Uses df_candidates as the pool for filtering and similarity matching.
    Only recommends if the best match similarity score >= similarity_threshold.
    """
    # Default return values
    predicted_category = "N/A"
    closest_vehicle_type = "No suitable vehicle found"
    max_similarity_score = -1.0 # Initialize score

    # --- Input Data Preparation ---
    try:
        input_product_type = input_data["Product Type"].values[0]
        input_purpose = input_data["Purpose"].values[0]
        input_weight = input_data[product_weight_column].values[0]

        # Create input features matching the training format
        input_encoded = encoder.transform([[input_product_type, input_purpose]])
        input_final = np.hstack((input_encoded, np.array([[input_weight]])))
    except Exception as e:
        print(f"Error processing input data: {e}")
        return predicted_category, closest_vehicle_type # Return defaults on input error

    # --- 1. Predict Category ---
    try:
        predicted_category = model_category.predict(input_final)[0]
    except Exception as e:
        print(f"Error predicting category for input {input_data.iloc[0]}: {e}")
        return predicted_category, closest_vehicle_type # Return defaults

    # --- 2. Filter candidates from the provided candidate pool (df_candidates) ---
    try:
        filtered_df = df_candidates[
            (df_candidates["Vehicle Category"] == predicted_category) &
            (df_candidates["maxWeightCapacity"] >= input_weight)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Apply strict purpose filter if needed
        if input_purpose in STRICT_PURPOSES:
            # Ensure the 'Purpose' column exists in filtered_df before filtering
            if 'Purpose' in filtered_df.columns:
                 filtered_df = filtered_df[filtered_df["Purpose"] == input_purpose]
            else:
                 print("Warning: 'Purpose' column missing in filtered candidates during strict filtering.")


        if not filtered_df.empty:
            # Encode the filtered candidates
            filtered_encoded = encoder.transform(filtered_df[["Product Type", "Purpose"]])
            # Important: Use the input weight for similarity calculation, not candidate weights
            # Create feature vectors for filtered candidates using *input weight*
            filtered_final = np.hstack((filtered_encoded, np.full((len(filtered_df), 1), input_weight)))

            # --- 3. Calculate Cosine Similarity ---
            try:
                similarity_scores = cosine_similarity(input_final, filtered_final)
                # Find the index and score of the most similar vehicle
                closest_idx = np.argmax(similarity_scores[0]) # Get index from the first (and only) row
                max_similarity_score = similarity_scores[0, closest_idx]

                # --- 4. Apply Threshold ---
                if max_similarity_score >= similarity_threshold:
                    closest_vehicle_type = filtered_df["Vehicle Type"].iloc[closest_idx]
                    # Optional: Print why it passed
                    # print(f"Input: {input_product_type}/{input_weight}kg -> Found: {closest_vehicle_type} (Score: {max_similarity_score:.4f} >= Threshold: {similarity_threshold})")
                else:
                    # Keep default "No suitable vehicle found" but maybe log why
                    # print(f"Input: {input_product_type}/{input_weight}kg -> Best match score {max_similarity_score:.4f} < Threshold {similarity_threshold}. No recommendation.")
                    closest_vehicle_type = "No suitable vehicle found (below threshold)" # More specific message


            except ValueError as e:
                print(f"Warning: Cosine similarity calculation error for input {input_data.iloc[0]}. Error: {e}")
                # Keep default "No suitable vehicle found"
            except IndexError as e:
                print(f"Warning: Indexing error after similarity calculation (likely empty similarity_scores). Input: {input_data.iloc[0]}. Error: {e}")
                # Keep default "No suitable vehicle found"

        # else: # No vehicles found after filtering (weight/category/purpose)
            # Optional: Log this case
            # print(f"Input: {input_product_type}/{input_weight}kg -> No vehicles found after initial filtering (Category: {predicted_category}, Purpose: {input_purpose}, Weight >= {input_weight}kg)")

    except KeyError as e:
        print(f"Error accessing columns during filtering for input {input_data.iloc[0]}. Missing column: {e}")
        # Keep default "No suitable vehicle found"
    except Exception as e:
        print(f"An unexpected error occurred during filtering/similarity for input {input_data.iloc[0]}: {e}")
        # Keep default "No suitable vehicle found"


    return predicted_category, closest_vehicle_type, max_similarity_score # Return score as well

# --- Define the Similarity Threshold ---
SIMILARITY_THRESHOLD = 0.80 # <<< ADJUST THIS VALUE (0.0 to 1.0)
print(f"\n--- Using Cosine Similarity Threshold: {SIMILARITY_THRESHOLD} ---")

# --- Evaluate on Test Set ---
predicted_categories = []
recommended_vehicle_types = []
similarity_scores_list = []

print("\n--- Evaluating on Test Set ---")
# Iterate through the test set samples
for i in range(len(df_test)):
    # Create input DataFrame for the current test sample
    test_sample = df_test.iloc[[i]]
    input_data = pd.DataFrame([{
        "Product Type": test_sample["Product Type"].values[0],
        "Product Weight (kg)": test_sample["Product Weight (kg)"].values[0],
        "Purpose": test_sample["Purpose"].values[0]
    }])

    # Get recommendation using the model trained on X_train and df_train as candidate pool
    pred_cat, rec_vehicle, sim_score = get_recommendation_eval_with_threshold(
        input_data,
        encoder,
        model_category,
        df_train, # Pass df_train here as the candidate pool
        similarity_threshold=SIMILARITY_THRESHOLD # Pass the threshold
    )

    predicted_categories.append(pred_cat)
    recommended_vehicle_types.append(rec_vehicle)
    similarity_scores_list.append(sim_score) # Store the similarity score

# --- Calculate and Print Metrics ---

# 1. Category Prediction Evaluation (Remains the same)
print("\n--- Vehicle Category Prediction Evaluation ---")
# Filter out N/A predictions if any occurred due to errors before prediction step
valid_indices_cat = [idx for idx, cat in enumerate(predicted_categories) if cat != "N/A"]
if not valid_indices_cat:
    print("No valid category predictions were made.")
else:
    y_category_test_filtered = y_category_test.iloc[valid_indices_cat]
    predicted_categories_filtered = [predicted_categories[i] for i in valid_indices_cat]

    try:
        category_accuracy = accuracy_score(y_category_test_filtered, predicted_categories_filtered)
        print(f"Accuracy: {category_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_category_test_filtered, predicted_categories_filtered, labels=model_category.classes_, zero_division=0))
        print("\nConfusion Matrix:")
        # Ensure labels used in confusion matrix are present in both true and predicted values
        unique_labels = np.unique(np.concatenate((y_category_test_filtered.unique(), np.unique(predicted_categories_filtered))))
        print(confusion_matrix(y_category_test_filtered, predicted_categories_filtered, labels=unique_labels))
    except Exception as e:
        print(f"Error calculating category metrics: {e}")


# 2. Final Vehicle Type Recommendation Evaluation (Accounts for threshold failures)
print(f"\n--- Full Recommendation (Vehicle Type) Evaluation (Threshold: {SIMILARITY_THRESHOLD}) ---")

# Treat "No suitable vehicle found..." as incorrect predictions if the actual test sample had a vehicle.
vehicle_type_accuracy = accuracy_score(y_vehicle_type_test, recommended_vehicle_types)
print(f"Overall Recommendation Accuracy (Exact Match): {vehicle_type_accuracy:.4f}")

# More detailed analysis
correct_recommendations = sum(1 for yt, yp in zip(y_vehicle_type_test, recommended_vehicle_types) if yt == yp)
no_vehicle_found_count = recommended_vehicle_types.count("No suitable vehicle found")
below_threshold_count = recommended_vehicle_types.count("No suitable vehicle found (below threshold)")
wrong_vehicle_count = len(df_test) - correct_recommendations - no_vehicle_found_count - below_threshold_count # Errors from category or filtering logic

print(f"\nBreakdown:")
print(f"- Correct Recommendations: {correct_recommendations} / {len(df_test)}")
print(f"- No Vehicle Found (Initial Filtering): {no_vehicle_found_count}")
print(f"- No Vehicle Found (Below Threshold {SIMILARITY_THRESHOLD}): {below_threshold_count}")
print(f"- Incorrect Vehicle Recommended (Wrong Type): {wrong_vehicle_count}") # This count might need adjustment based on how N/A categories were handled


# --- Example Usage (Optional - Keep your original example if needed) ---
print("\n--- Example Recommendation on New Input ---")
new_input_example = pd.DataFrame([{
    "Product Type": "Leafy Greens", #"Fruits",
    "Product Weight (kg)": 100, #3000,
    #"Purpose": "crops" # Let it be calculated
}])
# Ensure Purpose is added if not provided
if "Purpose" not in new_input_example.columns and "Product Type" in new_input_example.columns:
     new_input_example["Purpose"] = new_input_example["Product Type"].apply(map_purpose)

# Use the evaluation function, passing the TRAINING data as the candidate pool AND the threshold
recommended_category_ex, closest_vehicle_type_ex, sim_score_ex = get_recommendation_eval_with_threshold(
    new_input_example,
    encoder,
    model_category,
    df_train, # Use df_train as candidates
    similarity_threshold=SIMILARITY_THRESHOLD
)

# Output
print(f"\nInput: {new_input_example.iloc[0].to_dict()}")
print(f"Threshold Used: {SIMILARITY_THRESHOLD}")
print(f"✅ Recommended Vehicle Category: {recommended_category_ex}")
print(f"🚚 Closest Matching Vehicle Type: {closest_vehicle_type_ex}")
if closest_vehicle_type_ex not in ["No suitable vehicle found", "No suitable vehicle found (below threshold)"]:
    print(f"   (Similarity Score: {sim_score_ex:.4f})")
elif closest_vehicle_type_ex == "No suitable vehicle found (below threshold)":
     print(f"   (Highest Similarity Score Found: {sim_score_ex:.4f}, but below threshold)")