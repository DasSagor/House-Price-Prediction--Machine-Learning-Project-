# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import joblib # For saving/loading models and other Python objects
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

print("Starting model training script...")

# --- 1. Load Data ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found. Please ensure they are in the same directory.")
    exit()

# --- 2. Initial Setup: Handle IDs, Target Variable, Combine Data ---
train_ids = train_df['Id']
test_ids = test_df['Id']

# Drop Id column
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

# Log transform SalePrice (target variable)
y_train = np.log1p(train_df['SalePrice'])
train_df = train_df.drop('SalePrice', axis=1)

# Combine train and test data for consistent preprocessing
all_data = pd.concat([train_df, test_df]).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

# --- 3. Store Preprocessing Metadata (CRUCIAL for web app) ---
# Dictionary to store quality mappings
quality_map = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, # Common for general quality
    'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1 # For BsmtFinType
}

# List of columns to apply quality mapping
quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
                'BsmtFinType1', 'BsmtFinType2'] # BsmtFinType also has a scale

# --- 4. Feature Engineering & Preprocessing ---

# Impute Missing Values
print("Imputing missing values...")
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',
            'MasVnrType'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# For few remaining, fill with mode
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'Functional', 'SaleType', 'Utilities'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Feature Engineering
print("Performing feature engineering...")
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + \
                        all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['YearBuiltRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']
all_data['YearsSinceRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['YearsSinceBuilt'] = all_data['YrSold'] - all_data['YearBuilt']

# Handle potential negative values from Year calculations if YearSold < YearBuilt/RemodAdd
all_data.loc[all_data['YearsSinceRemod'] < 0, 'YearsSinceRemod'] = 0
all_data.loc[all_data['YearsSinceBuilt'] < 0, 'YearsSinceBuilt'] = 0

all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_data['OverallQual_OverallCond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['GrLivArea_OverallQual'] = all_data['GrLivArea'] * all_data['OverallQual']

# Apply quality mapping
print("Applying quality mappings...")
for col in quality_cols:
    all_data[col] = all_data[col].map(quality_map).fillna(0) # Fill any missing with 0 after mapping


# Log transform skewed numerical features
print("Transforming skewed features...")
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skew = skewed_feats[skewed_feats > 0.75]
skewed_features_to_transform = high_skew.index.tolist() # Convert to list for saving

for feat in skewed_features_to_transform:
    # Ensure the feature is not the target variable itself (though already split)
    if feat in all_data.columns:
        all_data[feat] = np.log1p(all_data[feat])

# One-Hot Encoding for remaining categorical features
print("Applying one-hot encoding...")
# Identify categorical columns that are still objects after quality mapping
remaining_categorical_cols = all_data.select_dtypes(include='object').columns.tolist()
all_data = pd.get_dummies(all_data, columns=remaining_categorical_cols, drop_first=True)

# Store final column names after all preprocessing
final_columns = all_data.columns.tolist()

# --- 5. Split Data back into Train and Test for Model Training ---
X_train = all_data.iloc[:len(y_train)]
X_test = all_data.iloc[len(y_train):] # This will be used only for getting columns in the web app

print(f"X_train shape after preprocessing: {X_train.shape}")
print(f"X_test shape after preprocessing: {X_test.shape}")

# --- 6. Train the Model ---
print("Training the XGBoost model...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Using optimized parameters (you can tune these further if you wish)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=3000,
                             learning_rate=0.01,
                             max_depth=4,
                             subsample=0.7,
                             colsample_bytree=0.7,
                             random_state=42,
                             n_jobs=-1)

xgb_model.fit(X_train, y_train)
print("Model training complete.")

# --- 7. Save Model and Preprocessing Components ---
print("Saving model and preprocessing data...")
joblib.dump(xgb_model, 'house_price_model.joblib')

# Save all metadata needed to preprocess new input consistently
preprocessing_metadata = {
    'final_columns': final_columns,
    'quality_map': quality_map,
    'quality_cols': quality_cols,
    'skewed_features_to_transform': skewed_features_to_transform,
    'categorical_features_for_onehot': remaining_categorical_cols, # The columns that were one-hot encoded
    # Add any specific imputation values if you used mean/median for numericals, e.g.:
    # 'lotfrontage_median': train_df['LotFrontage'].median() # if not groupby
}
joblib.dump(preprocessing_metadata, 'preprocessing_metadata.joblib')

print("Model and preprocessing metadata saved successfully!")
print("Now run 'python app.py' to start the web application.")