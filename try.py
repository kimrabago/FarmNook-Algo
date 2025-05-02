import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Good practice, though we'll train on all data here for simplicity
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Data Loading and Preprocessing ---

# Load dataset
df = pd.read_csv("farmnook_dataset_v2.csv")

# Define purpose mapping
def map_purpose(product_type):
    product = str(product_type).lower() # Ensure input is string
    if product in ["pigs", "cows", "goats", "chickens"]:
        return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]:
        return "perishable goods"
    # Consider specific crop types if needed, otherwise default to "crops"
    elif product in ["corn", "rice", "fruit", "wheat", "vegetables"]: # Example crops
         return "crops"
    else:
        # Default assumption or handle unknown types
        # print(f"Warning: Product type '{product_type}' not explicitly mapped, defaulting to 'crops'.")
        return "crops" # Defaulting unknowns to crops

# Add Purpose column if missing
if "Purpose" not in df.columns:
    # Ensure 'Product Type' exists before applying the map
    if "Product Type" in df.columns:
        df["Purpose"] = df["Product Type"].apply(map_purpose)
    else:
        raise ValueError("Dataset must contain a 'Product Type' column.")

# --- Feature Engineering and Model Training ---

# Define features (X) and target (y)
# Make sure necessary columns exist
required_cols = ["Product Type", "Purpose", "Product Weight (kg)", "Vehicle Type"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

# Drop rows with missing values in crucial columns to avoid errors during training/encoding
df.dropna(subset=required_cols, inplace=True)
df = df[df['Product Weight (kg)'] > 0] # Ensure weight is positive
df = df[df['Vehicle Type'].notna() & (df['Vehicle Type'] != '')] # Ensure target is valid

if df.empty:
     raise ValueError("Dataset is empty after cleaning. Cannot train the model.")


features = ["Product Type", "Purpose", "Product Weight (kg)"]
target = "Vehicle Type"

X = df[features]
y = df[target]

# Define preprocessing steps
# OneHotEncode categorical features, pass through numerical features
# handle_unknown='ignore' is crucial for predicting on new data with unseen categories
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Product Type', 'Purpose']),
        ('num', 'passthrough', ['Product Weight (kg)']) # Keep weight as is
    ],
    remainder='drop' # Drop other columns if any
)

# Create the full pipeline with preprocessing and the Decision Tree model
# We use DecisionTreeClassifier because Vehicle Type is a category.
# Set max_depth to prevent potential overfitting on complex datasets. Tune as needed.
# Set min_samples_leaf to ensure leaves aren't too specific. Tune as needed.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5))
])

# Train the model
# For simplicity here, we train on the whole dataset.
# In practice, use train_test_split for evaluation.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model_pipeline.fit(X_train, y_train)
# print(f"Model Accuracy on Test Set: {model_pipeline.score(X_test, y_test):.4f}") # Example evaluation

try:
    model_pipeline.fit(X, y)
    print("Model training complete.")
except ValueError as e:
    print(f"Error during model training: {e}")
    print("This might happen if the dataset is too small or lacks variety after cleaning.")
    # Optional: exit or handle appropriately
    exit()


# --- Recommendation Function ---

def get_recommendation_dt(input_data, pipeline_model, top_n=5):
    """
    Recommends vehicle types based on input using a trained Decision Tree pipeline.

    Args:
        input_data (pd.DataFrame): DataFrame with input features
                                   (must include columns used for training).
        pipeline_model (Pipeline): The trained scikit-learn pipeline object.
        top_n (int): Maximum number of recommendations to return.

    Returns:
        list: A list of recommended vehicle types, sorted by probability.
              Returns ["No suitable vehicle found based on learned data"] if no prediction.
    """
    if not hasattr(pipeline_model.named_steps['classifier'], 'classes_'):
         return ["Model not trained yet or training failed."]

    try:
        # Predict probabilities for all classes
        probabilities = pipeline_model.predict_proba(input_data)

        # Get class labels (vehicle types) in the order of probability columns
        vehicle_classes = pipeline_model.named_steps['classifier'].classes_

        # Create a list of (vehicle_type, probability) tuples for the first input sample
        # Assumes input_data has only one row for prediction
        prob_list = list(zip(vehicle_classes, probabilities[0]))

        # Sort by probability in descending order
        prob_list.sort(key=lambda x: x[1], reverse=True)

        # Filter out vehicles with zero probability and get the top N
        recommended_vehicles = [vehicle for vehicle, prob in prob_list if prob > 0][:top_n]

        if not recommended_vehicles:
            # This might happen if the input falls into a tree region where no vehicle was seen
            # during training, or if handle_unknown='ignore' results in all zero probabilities
            # for completely unseen categorical combinations.
            return ["No suitable vehicle found based on learned data"]

        return recommended_vehicles

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Check if the error is due to unseen features if handle_unknown wasn't 'ignore' or structure mismatch
        return ["Error occurred during recommendation"]


# --- Example Usage ---

# Ensure the input DataFrame has the same columns as used for training X
# And apply the same preprocessing logic (like mapping purpose)
new_product_type = "cow"
new_weight = 600
new_purpose = "livestock" # Derive purpose

new_input_df = pd.DataFrame([{
    "Product Type": new_product_type,
    "Product Weight (kg)": new_weight,
    "Purpose": new_purpose
    # Ensure the column order matches the 'features' list if not using ColumnTransformer names explicitly
}])


# Get recommendations using the Decision Tree model
recommended_vehicle_types_dt = get_recommendation_dt(new_input_df, model_pipeline, top_n=3) # Get top 3

# Output
print("\n--- Decision Tree Recommendation ---")
print(f"Input: Product='{new_product_type}', Weight={new_weight}kg, Purpose='{new_purpose}'")
print("🚚 Recommended Vehicle Types (most likely first):")
if recommended_vehicle_types_dt:
    for i, v in enumerate(recommended_vehicle_types_dt):
        print(f"{i+1}. {v}")
else:
    print("- No suitable vehicle found based on learned data.")