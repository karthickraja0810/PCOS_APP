# --------------------------------------------
# PCOS Prediction Model Training (XGBoost)
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------
df = pd.read_csv(r"C:\Users\Karthickraja.S\Downloads\Data_for_ML (1).xls")
print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------------------------------------
# 2. Clean and Preprocess Data
# -------------------------------------------------------

# Drop irrelevant columns
df = df.drop(['Sl. No', 'Patient File No.'], axis=1, errors='ignore')

# Remove unnamed or empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Replace special characters and clean data
df = df.replace(['?', 'NA', 'N/A', '--', ' ', 'nan', 'Nan'], np.nan)
df = df.applymap(lambda x: str(x).replace('..', '.').replace(',', '.').strip() if isinstance(x, str) else x)

# CRITICAL FIX: Explicitly handle data types and clean Blood Group
for col in df.columns:
    if col == 'PCOS (Y/N)':
        # Convert target to numeric
        df[col] = pd.to_numeric(df[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}), errors='coerce').fillna(0).astype(int)
    elif col == 'Blood Group':
        # --- ENSURE BLOOD GROUP IS STRING/OBJECT ---
        # Keep Blood Group as a string/object type.
        df[col] = df[col].astype(str).str.upper().str.strip().replace({'NAN': np.nan, '0': np.nan})
        # If your data uses numeric codes (like 12, 13) for blood groups, they will now be treated as strings.
    else:
        # Force all other features (labs, vitals) to numeric, coercing bad strings to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
# 1. Fill missing numeric values with median
for col in df.select_dtypes(include=[np.number]).columns:
    # Exclude the target column if it's still present
    if col != 'PCOS (Y/N)': 
        df[col] = df[col].fillna(df[col].median())

# 2. Fill missing categorical values (Blood Group) with the most frequent category
if 'Blood Group' in df.columns and df['Blood Group'].dtype == 'object':
    df['Blood Group'] = df['Blood Group'].fillna(df['Blood Group'].mode()[0])

# 3. Encode Binary Columns (Y/N, Yes/No)
binary_cols = [
    'Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
    'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
    'Fast food (Y/N)', 'Reg.Exercise(Y/N)'
]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}).fillna(0)
# Encode target column
if 'PCOS (Y/N)' in df.columns:
    df['PCOS (Y/N)'] = df['PCOS (Y/N)'].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0})
else:
    raise ValueError("❌ Target column 'PCOS (Y/N)' not found in dataset!")

# Fill any remaining missing categorical values
df = df.fillna(0)

# -------------------------------------------------------
# 4. Feature and Target Split
# -------------------------------------------------------
X = df.drop(['PCOS (Y/N)'], axis=1)
y = df['PCOS (Y/N)']

# Convert any non-numeric columns using get_dummies
X = pd.get_dummies(X, drop_first=True)
print("\nFinal Model Feature List:")
print(X.columns.tolist())
print(f"Total final features used for training: {X.shape[1]}")

print("✅ After Encoding: X shape =", X.shape)

# -------------------------------------------------------
# 5. Feature Scaling
# -------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------
# 6. Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# 7. Train XGBoost Model
# -------------------------------------------------------
model = XGBClassifier(
    n_estimators=350,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_lambda=1,
    reg_alpha=0.5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -------------------------------------------------------
# 8. Evaluate the Model
# -------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------------
# 9. Save Model and Scaler
# -------------------------------------------------------
joblib.dump(model, "pcos_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n🎉 Model and Scaler saved successfully!")
