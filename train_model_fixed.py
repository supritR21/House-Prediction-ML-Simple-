import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. LOAD AND PREPROCESS DATA ======================
print("Step 1: Loading and preprocessing data...")
data = pd.read_csv('mumbaiproject.csv')

# dropping the columns which are not needed
data = data.drop(['zaroni', 'status', 'society', 'transaction'], axis=1)

# extracting size from location
data = data[data['location'].str.contains("Studio|Plot") == False]
data['size'] = data['location'].str[0:2]
data = data[data['size'].str.contains("Ap|>|Ho|Vi") == False]
data['size'] = data['size'].astype(int)

# extracting just the location from column
new = data["location"].str.split("in ", n=1, expand=True)
data['loc'] = new[1]
data = data.drop(['location'], axis=1)

# converting parking to int
data['parking'] = data['parking'].str[0:2]
data['parking'] = data['parking'].fillna(1)
data['parking'] = data['parking'].astype(int)

# converting bath to int
data['bath'] = data['bath'].str[0:2]
data['bath'] = data['bath'].fillna(1)
data = data[data['bath'].str.contains(">") == False]
data['bath'] = data['bath'].astype(int)

def convert_price(price):
    try:
        amount, unit = price.split()
        amount = float(amount)
        if unit == 'Cr':
            amount *= 10 ** 7
        elif unit == 'Lac':
            amount *= 10 ** 5
        return int(round(amount))
    except ValueError:
        return None

data['price'] = data['price'].apply(convert_price)
data['price'] = data['price'].fillna(data['price'].mean())
data['price'] = data['price'].astype(int)

new1 = data["total_sqft"].str.split("sq", n=1, expand=True)
data['sqft'] = new1[0]
data['sqft'] = data['sqft'].fillna(data['sqft'].median())
data['sqft'] = data['sqft'].astype(int)
data = data.drop(['total_sqft'], axis=1)

data['furnishing'] = data['furnishing'].fillna('Furnished')

# ====================== 2. DATA VALIDATION ======================
print("\nStep 2: Data validation and cleaning...")

# Check initial data statistics
print(f"Initial data shape: {data.shape}")
print(f"Initial price range: ₹{data['price'].min():,} to ₹{data['price'].max():,}")

# Remove any rows with missing values
data = data.dropna()
print(f"After removing NaN: {data.shape}")

# Ensure all numeric values are positive
data = data[(data['price'] > 0) & 
            (data['size'] > 0) & 
            (data['sqft'] > 0) & 
            (data['bath'] > 0) & 
            (data['parking'] >= 0)]

print(f"After removing non-positive values: {data.shape}")

# ====================== 3. OUTLIER REMOVAL ======================
print("\nStep 3: Removing outliers...")

# Remove outliers using IQR method (more robust than quantile)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.05)  # 5th percentile
    Q3 = df[column].quantile(0.95)  # 95th percentile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply to each numeric column
numeric_cols = ['price', 'sqft', 'size', 'bath', 'parking']
for col in numeric_cols:
    initial_len = len(data)
    data = remove_outliers_iqr(data, col)
    removed = initial_len - len(data)
    if removed > 0:
        print(f"  Removed {removed} outliers from {col}")

print(f"After outlier removal: {data.shape}")

# ====================== 4. DATA STATISTICS ======================
print("\nStep 4: Data statistics after preprocessing:")
print(f"Price range: ₹{data['price'].min():,} to ₹{data['price'].max():,}")
print(f"Size (BHK) range: {data['size'].min()} to {data['size'].max()}")
print(f"Sqft range: {data['sqft'].min()} to {data['sqft'].max()}")
print(f"Bathroom range: {data['bath'].min()} to {data['bath'].max()}")
print(f"Parking range: {data['parking'].min()} to {data['parking'].max()}")

print("\nUnique locations:", len(data['loc'].unique()))
print("Unique furnishing types:", data['furnishing'].unique())

print("\nSample of processed data:")
print(data.head())

# Save the processed data for visualization
data.to_csv('finaldata_processed.csv', index=False)
print("\nProcessed data saved as 'finaldata_processed.csv'")

# ====================== 5. MODEL TRAINING ======================
print("\nStep 5: Training the model...")

X = data.drop(['price'], axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Create preprocessing pipeline
column_transform = make_column_transformer(
    (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['loc', 'furnishing']),
    remainder='passthrough'
)

scaler = StandardScaler()

# Try both LinearRegression and Ridge for comparison
print("\nTraining models...")

# Model 1: Linear Regression
lr_model = LinearRegression()
pipe_lr = make_pipeline(column_transform, scaler, lr_model)
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

# Model 2: Ridge Regression (better for preventing extreme values)
ridge_model = Ridge(alpha=1.0, random_state=42)
pipe_ridge = make_pipeline(column_transform, scaler, ridge_model)
pipe_ridge.fit(X_train, y_train)
y_pred_ridge = pipe_ridge.predict(X_test)

# ====================== 6. MODEL EVALUATION ======================
print("\nStep 6: Model Evaluation")
print("-" * 50)

# Linear Regression metrics
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print(f"Linear Regression:")
print(f"  R² Score: {r2_lr:.4f}")
print(f"  MAPE: {mape_lr:.4f}")
print(f"  Min Prediction: ₹{y_pred_lr.min():,.0f}")
print(f"  Max Prediction: ₹{y_pred_lr.max():,.0f}")

# Ridge Regression metrics
r2_ridge = r2_score(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)
print(f"\nRidge Regression:")
print(f"  R² Score: {r2_ridge:.4f}")
print(f"  MAPE: {mape_ridge:.4f}")
print(f"  Min Prediction: ₹{y_pred_ridge.min():,.0f}")
print(f"  Max Prediction: ₹{y_pred_ridge.max():,.0f}")

# Check for negative predictions
negative_lr = np.sum(y_pred_lr < 0)
negative_ridge = np.sum(y_pred_ridge < 0)
print(f"\nNegative predictions check:")
print(f"  Linear Regression: {negative_lr} negative predictions")
print(f"  Ridge Regression: {negative_ridge} negative predictions")

# ====================== 7. SAVE THE BEST MODEL ======================
print("\nStep 7: Saving the best model...")

# Choose Ridge model if it has no negative predictions and similar accuracy
if negative_ridge == 0 and r2_ridge > 0.7:
    print("✓ Saving Ridge Regression model (no negative predictions)")
    joblib.dump(pipe_ridge, 'work_new.joblib')
    best_model = pipe_ridge
else:
    print("✓ Saving Linear Regression model")
    joblib.dump(pipe_lr, 'work_new.joblib')
    best_model = pipe_lr

print("Model saved as 'work_new.joblib'")

# ====================== 8. TEST PREDICTIONS ======================
print("\nStep 8: Sample test predictions:")
print("-" * 50)

# Test with some sample inputs
test_samples = [
    ['Andheri East Mumbai', 'Furnished', 1000, 2, 2, 1],
    ['Borivali Mumbai', 'Semi-Furnished', 800, 1, 1, 1],
    ['Kandivali West Mumbai', 'Unfurnished', 1200, 3, 2, 2]
]

test_df = pd.DataFrame(test_samples, columns=['loc', 'furnishing', 'sqft', 'size', 'bath', 'parking'])

print("Sample predictions:")
for i, (_, row) in enumerate(test_df.iterrows()):
    pred = best_model.predict(pd.DataFrame([row]))[0]
    print(f"\nSample {i+1}:")
    print(f"  Location: {row['loc']}")
    print(f"  Furnishing: {row['furnishing']}")
    print(f"  Area: {row['sqft']} sqft, BHK: {row['size']}")
    print(f"  Bathrooms: {row['bath']}, Parking: {row['parking']}")
    print(f"  Predicted Price: ₹{pred:,.0f}")
    
    if pred <= 0:
        print("  ⚠️ WARNING: Negative/zero prediction!")

# ====================== 9. FEATURE IMPORTANCE ======================
print("\nStep 9: Feature Analysis")
print("-" * 50)

# Get feature names after one-hot encoding
if hasattr(best_model, 'named_steps'):
    # For pipeline
    col_transformer = best_model.named_steps['columntransformer']
    ohe = col_transformer.named_transformers_['onehotencoder']
    feature_names = ohe.get_feature_names_out(['loc', 'furnishing'])
    numerical_features = ['sqft', 'size', 'bath', 'parking']
    all_features = list(feature_names) + numerical_features
    
    if hasattr(best_model.named_steps['ridge'], 'coef_'):
        coefficients = best_model.named_steps['ridge'].coef_
    elif hasattr(best_model.named_steps['linearregression'], 'coef_'):
        coefficients = best_model.named_steps['linearregression'].coef_
    
    # Show top 10 most important features
    if 'coefficients' in locals():
        feature_importance = pd.DataFrame({
            'feature': all_features,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nTop 10 most influential features:")
        print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nNext steps:")
print("1. Update your Streamlit app to load 'work_new.joblib'")
print("2. Test the new model with various inputs")
print("3. Check that all predictions are positive and realistic")