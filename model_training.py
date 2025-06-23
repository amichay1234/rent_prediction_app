from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# ========== Step 1: Feature Engineering ==========
# Create a copy of the dataset with selected features
df_selected = my_data_test[features].copy()

# ========== Step 2: Define X and y ==========
X = df_selected.drop("price", axis=1)  # Features
y = df_selected["price"]              # Target variable

# ========== Step 3: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Step 4: Define Transformers ==========
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

# Define transformer for categorical features (imputation + encoding)
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Define transformer for numeric features (imputation + scaling)
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Combine transformers into a column transformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ========== Step 5: Create Pipeline with ElasticNet ==========
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.3))  # Sample hyperparameters
])

# ========== Step 6: Cross Validation ==========
# Perform 10-fold CV using negative MAE as the scoring metric
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring="neg_mean_absolute_error")
mean_cv_mae = -np.mean(cv_scores)  # Convert negative to positive MAE

# ========== Step 7: Train the Model ==========
pipeline.fit(X_train, y_train)

# ========== Step 8: Evaluate Model Performance ==========
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"📊 MAE: {mae:.2f} ₪")
print(f"📉 RMSE: {rmse:.2f} ₪")
print(f"🔢 R²: {r2:.4f}")
print(f"📚 10-Fold CV MAE: {mean_cv_mae:.2f} ₪")

# ========== Step 9: Feature Importance (ElasticNet Coefficients) ==========
# Retrieve transformed feature names
cat_features = pipeline.named_steps["preprocessor"].named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols)
num_features = numeric_cols
all_features = np.concatenate([num_features, cat_features])

# Extract model coefficients
coefficients = pipeline.named_steps["regressor"].coef_

# Create a Series mapping features to their coefficients
feature_importance = pd.Series(coefficients, index=all_features)

# Sort by absolute importance and show top 5
top5 = feature_importance.reindex(feature_importance.abs().sort_values(ascending=False).index).head(5)
print("\n🔥 Top 5 Influential Features:")
print(top5)
