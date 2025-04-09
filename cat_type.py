from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("ph_vehicle_dataset_strict_corrected.csv")

# Define purpose mapping
def map_purpose(product_type):
    product = product_type.lower()
    if product in ["pigs", "cows", "goats", "chickens"]:
        return "livestock"
    elif product in ["frozen fish", "meat", "milk", "fresh vegetables", "leafy greens", "eggs"]:
        return "perishable crops"
    else:
        return "crops"

# Add Purpose column if missing
if "Purpose" not in df.columns:
    df["Purpose"] = df["Product Type"].apply(map_purpose)

# Ensure vehicle category is present
category_map = {
    "Multicab": "N1", "Tricycle": "L1", "Delivery Van": "N1",
    "Elf Truck": "N2", "Refrigerated Truck": "N2", "Livestock Truck": "N2",
    "Pickup Truck": "N1", "Open Truck": "N1", "Refrigerated Van": "N1",
    "Mini Livestock Truck": "N1", "10-Wheeler Truck": "N3",
    "Heavy Livestock Trailer": "N3", "Container Truck": "N3"
}
if "Vehicle Category" not in df.columns:
    df["Vehicle Category"] = df["Vehicle Type"].map(category_map).fillna("N1")

# One-hot encode
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[["Product Type", "Purpose"]])
X_final = np.hstack((X_encoded, df[["Product Weight (kg)"]].values))

# Align shapes
min_len = min(len(X_final), len(df))
X_final = X_final[:min_len]
df = df.iloc[:min_len].reset_index(drop=True)

# Train Decision Tree model for category
model_category = DecisionTreeClassifier()
model_category.fit(X_final, df["Vehicle Category"])

# Strict filtering enabled for all purpose types
STRICT_PURPOSES = ["livestock", "perishable crops", "crops"]

# Get prediction and best vehicle type
def get_recommendation(input_data, trained_data, encoder, product_column="Product Weight (kg)"):
    input_encoded = encoder.transform(input_data.drop(columns=[product_column]))
    input_final = np.hstack((input_encoded, np.array([[input_data[product_column].values[0]]])))

    predicted_category = model_category.predict(input_final)[0]
    purpose = input_data["Purpose"].values[0]
    weight = input_data[product_column].values[0]

    # Filter candidates from dataset
    filtered_df = df[
        (df["Vehicle Category"] == predicted_category) &
        (df["maxWeightCapacity"] >= weight)
    ]

    # Apply strict purpose filter
    if purpose in STRICT_PURPOSES:
        filtered_df = filtered_df[filtered_df["Purpose"] == purpose]

    if not filtered_df.empty:
        filtered_encoded = encoder.transform(filtered_df[["Product Type", "Purpose"]])
        filtered_final = np.hstack((filtered_encoded, filtered_df[[product_column]].values))
        similarity_scores = cosine_similarity(input_final, filtered_final)
        closest_idx = np.argmax(similarity_scores)
        closest_vehicle_type = filtered_df["Vehicle Type"].iloc[closest_idx]
    else:
        closest_vehicle_type = "No suitable vehicle found"

    return predicted_category, closest_vehicle_type

# Example usage, FARMER INPUTS HERE
new_input = pd.DataFrame([{
    "Product Type": "Ducks",
    "Product Weight (kg)": 3000,
    "Purpose": "livestock"
}])

# Get recommendation
recommended_category, closest_vehicle_type = get_recommendation(new_input, X_final, encoder)

# Output
print(f"✅ Recommended Vehicle Category: {recommended_category}")
print(f"🚚 Closest Matching Vehicle Type: {closest_vehicle_type}")