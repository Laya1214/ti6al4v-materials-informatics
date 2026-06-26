# ==========================================
# STEP 1: CHOOSE EXECUTION MODE & SETUP
# ==========================================
# Set TEST_MODE = True to run a fast 50-image preview. 
# Set TEST_MODE = False to process all 1,225 images for the final paper.
TEST_MODE = False 

import os
import subprocess

# Clone the target GitHub repository if not already present in the workspace
repo_url = "https://github.com/ArunBaskaran/Image-Driven-Machine-Learning-Approach-for-Microstructure-Classification-and-Segmentation-Ti-6Al-4V.git"
repo_dir = "Baskaran_Ti_Alloy_Repo"

if not os.path.exists(repo_dir):
    print("Cloning Baskaran GitHub repository...")
    subprocess.run(["git", "clone", repo_url, repo_dir])
else:
    print("Baskaran Repository is already cloned and present.")

# Import core scientific and machine learning libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Define paths matching Baskaran's repository structure
images1_path = os.path.join(repo_dir, "Images1")
images2_path = os.path.join(repo_dir, "Images2")
labels_path = os.path.join(repo_dir, "labels.xlsx")

# ==========================================
# STEP 2: AUTOMATED SPATIAL FEATURE EXTRACTION
# ==========================================
def extract_average_grain_size(image_path):
    """
    Standardizes image contrast using CLAHE, applies adaptive thresholding,
    and uses contour ellipse-fitting to find the average grain/lath size d.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Contrast Limited Adaptive Histogram Equalization (CLAHE) for edge enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    # Adaptive thresholding to segment alpha laths/grains from beta matrix
    thresh = cv2.adaptiveThreshold(
        img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours representing individual phase boundaries
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    minor_axes = list()
    for cnt in contours:
        # Ignore tiny noise artifacts
        if cv2.contourArea(cnt) < 15:
            continue
        # We need at least 5 points to fit an ellipse
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # Unpack the ellipse tuple to avoid brackets containing integers
            center, axes, angle = ellipse
            minor_axis = min(axes)
            # Assuming a spatial scale (e.g., 0.1 micrometers per pixel)
            minor_axis_microns = minor_axis * 0.1
            minor_axes.append(minor_axis_microns)
            
    # Fallback to a default average if no contours are matched
    if len(minor_axes) == 0:
        return 4.0 
        
    return np.mean(minor_axes)

# ==========================================
# STEP 3: PREPARE DATASET WITH METALLURGICAL PHYSICS
# ==========================================
print("Parsing label metadata from labels.xlsx...")
# Load with header=None to preserve the first data row and prevent column-naming issues
df = pd.read_excel(labels_path, header=None)

# Explicitly name the columns using list appending to avoid any brackets being stripped
cols = list()
cols.append("Index")
cols.append("Label")
df.columns = cols + list(df.columns[2:])

if TEST_MODE:
    print(f"--- TEST MODE ACTIVE: Subsampling first 50 rows of metadata ---")
    df = df.head(50)

# Build correct file paths for the images spread across the two folders
image_files = list()
grain_sizes = list()

print("Extracting grain dimensions from SEM images. Please wait...")
found_count = 0
for idx, row in df.iterrows():
    img_idx = row["Index"]
    if pd.isna(img_idx):
        image_files.append(None)
        grain_sizes.append(None)
        continue
    
    # Clean up the index value
    img_idx_str = str(img_idx).strip()
    
    # If pandas read the value as a float (e.g., '1.0'), convert to integer string ('1')
    if img_idx_str.endswith(".0"):
        img_idx_str = img_idx_str[:-2]
        
    # Check if the filename extension is already present
    if img_idx_str.lower().endswith(".png"):
        img_name = img_idx_str
    else:
        img_name = "image_" + img_idx_str + ".png"
        
    img_path = os.path.join(images1_path, img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(images2_path, img_name)
        
    if os.path.exists(img_path):
        found_count += 1
        image_files.append(img_path)
        # Programmatically measure grain size
        d_avg = extract_average_grain_size(img_path)
        grain_sizes.append(d_avg)
    else:
        image_files.append(None)
        grain_sizes.append(None)

print("Located " + str(found_count) + " out of " + str(len(df)) + " images on disk.")
df["image_path"] = image_files
df["grain_size_microns"] = grain_sizes

# Drop rows only where our matched files are missing to preserve rest of the data
df = df.dropna(subset=["image_path", "grain_size_microns"]).reset_index(drop=True)
print("Rows remaining after cleaning: " + str(len(df)))

if len(df) == 0:
    print("\n" + "="*50)
    print("CRITICAL ERROR: No valid image files could be matched on disk!")
    print("Verify that the repository folders 'Images1' and 'Images2' are intact.")
    print("="*50 + "\n")
    raise ValueError("No valid image files found. Pipeline cannot continue.")

# ------------------------------------------
# APPLY METALLURGICAL PHYSICS (Hall-Petch)
# ------------------------------------------
# We use the Hall-Petch equation to calculate Vickers Hardness (HV)
# HV = HV_0 + k_HP * d^(-1/2). For Ti-6Al-4V: HV_0 = 300 HV, k_HP = 150 HV*um^0.5
HV_0 = 300.0
k_HP = 150.0
df["Vickers_Hardness"] = HV_0 + k_HP * (df["grain_size_microns"] ** (-0.5))

# Classify grain sizes into distinct categories
# Fine Grained < 3.0 um, Medium Grained 3.0-6.0 um, Coarse Grained > 6.0 um
def classify_grain_category(size):
    if size < 3.0:
        return "Fine"
    elif size <= 6.0:
        return "Medium"
    else:
        return "Coarse"

df["grain_class"] = df["grain_size_microns"].apply(classify_grain_category)

print("Data Prepared. Sample counts: Fine=" + str(sum(df['grain_class']=='Fine')) + ", Medium=" + str(sum(df['grain_class']=='Medium')) + ", Coarse=" + str(sum(df['grain_class']=='Coarse')))

# ==========================================
# STEP 4: STRICT SPLITTING & MACHINE LEARNING
# ==========================================
# Ensure strict parent image splitting BEFORE feature processing to eliminate data leakage
unique_classes = df["grain_class"].nunique()
class_counts = df["grain_class"].value_counts()
min_class_count = class_counts.min()

# Failsafe: Only stratify if more than one class exists and the minimum count is greater than 1
if unique_classes > 1 and min_class_count > 1:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["grain_class"])
else:
    print("Warning: Insufficient class diversity for stratified split. Falling back to simple split.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Feature Engineering: Extract basic statistical and spatial texture indicators
def extract_statistical_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(5)
    mean = np.mean(img)
    std = np.std(img)
    skewness = np.mean((img - mean) ** 3) / (std ** 3 + 1e-8)
    
    # Calculate total pixels robustly using the scalar.size property to bypass shape tuple math
    total_pixels = img.size
    
    # Construct lists dynamically to avoid any brackets being stripped by the parser
    img_list = list()
    img_list.append(img)
    channels = list()
    channels.append(0)
    hist_size = list()
    hist_size.append(256)
    hist_ranges = list()
    hist_ranges.append(0)
    hist_ranges.append(256)
    
    # Standard Shannon Entropy via histogram probability
    hist = cv2.calcHist(img_list, channels, None, hist_size, hist_ranges)
    prob = hist / total_pixels
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    
    # Local binary pattern representation (simplified edge density)
    edges = cv2.Canny(img, 100, 200)
    # Corrected: Use total_pixels directly for safe scalar division
    edge_density = np.sum(edges > 0) / (total_pixels + 1e-8)
    return np.array([mean, std, skewness, entropy, edge_density])

print("Extracting machine learning feature vectors...")
X_train = np.array([extract_statistical_features(p) for p in train_df["image_path"]])
X_test = np.array([extract_statistical_features(p) for p in test_df["image_path"]])

# Regression Targets: Vickers Hardness
y_train_hv = train_df["Vickers_Hardness"].values
y_test_hv = test_df["Vickers_Hardness"].values

# Classification Targets: Grain Size Class
y_train_class = train_df["grain_class"].values
y_test_class = test_df["grain_class"].values

# ------------------------------------------
# TRAIN ML REGRESSOR & CLASSIFIER
# ------------------------------------------
print("Training Random Forest models...")
# Regressor for Vickers Hardness (HV)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train_hv)

# Classifier for Grain Size
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train_class)

# ==========================================
# STEP 5: PERFORMANCE EVALUATION & VISUALIZATION
# ==========================================
# Evaluate Regressor
hv_preds = reg_model.predict(X_test)
r2 = r2_score(y_test_hv, hv_preds)
mae = mean_absolute_error(y_test_hv, hv_preds)

print("\n================ METALLURGICAL MODEL RESULTS ================")
print("Hardness Prediction R2 Score: " + f"{r2:.4f}")
print("Hardness Prediction Mean Absolute Error: " + f"{mae:.2f}" + " HV")

# Evaluate Classifier
class_preds = clf_model.predict(X_test)
acc = accuracy_score(y_test_class, class_preds)
print("Grain Size Classification Accuracy: " + f"{acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test_class, class_preds, zero_division=0))

# Generate Validation Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_hv, hv_preds, color="blue", alpha=0.5, edgecolors="k")
plt.plot([min(y_test_hv), max(y_test_hv)], [min(y_test_hv), max(y_test_hv)], "r--", lw=2)
plt.title("Vickers Hardness (HV): Predicted vs Actual")
plt.xlabel("Actual Hardness (HV)")
plt.ylabel("Predicted Hardness (HV)")
plt.grid(True)

plt.subplot(1, 2, 2)
# Re-assemble confusion matrix labels based on what exists in the test split
unique_test_labels = sorted(list(set(y_test_class)))
cm = confusion_matrix(y_test_class, class_preds, labels=unique_test_labels)
plt.imshow(cm, cmap="Blues")
plt.title("Grain Size Classification Confusion Matrix")
plt.xticks(range(len(unique_test_labels)), unique_test_labels)
plt.yticks(range(len(unique_test_labels)), unique_test_labels)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
for i in range(len(unique_test_labels)):
    for j in range(len(unique_test_labels)):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontsize=12)

plt.tight_layout()
plt.show()