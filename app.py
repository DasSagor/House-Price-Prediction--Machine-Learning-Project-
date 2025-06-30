from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew

app = Flask(__name__)

# Load Model and Metadata
try:
    model = joblib.load('house_price_model.joblib')
    metadata = joblib.load('preprocessing_metadata.joblib')

    final_columns = metadata['final_columns']
    quality_map = metadata['quality_map']
    quality_cols = metadata['quality_cols']
    skewed_features_to_transform = metadata['skewed_features_to_transform']
    categorical_features_for_onehot = metadata['categorical_features_for_onehot']

    print("Model and preprocessing metadata loaded successfully.")
except FileNotFoundError:
    print("Error: Model or preprocessing metadata files not found.")
    exit()
except Exception as e:
    print(f"Error loading model or metadata: {e}")
    exit()

INPUT_FEATURES = {
    'OverallQual': {'type': 'select', 'options': list(range(1, 11))},
    'GrLivArea': {'type': 'number'},
    'GarageCars': {'type': 'number'},
    'GarageArea': {'type': 'number'},
    'TotalBsmtSF': {'type': 'number'},
    '1stFlrSF': {'type': 'number'},
    'FullBath': {'type': 'number'},
    'YearBuilt': {'type': 'number'},
    'YearRemodAdd': {'type': 'number'},
    'Fireplaces': {'type': 'number'},
    'TotRmsAbvGrd': {'type': 'number'},
    'LotArea': {'type': 'number'},
    'Neighborhood': {'type': 'select', 'options': ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste']},
    'HouseStyle': {'type': 'select', 'options': ['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin']},
    'ExterQual': {'type': 'select', 'options': ['Ex', 'Gd', 'TA', 'Fa']},
    'KitchenQual': {'type': 'select', 'options': ['Ex', 'Gd', 'TA', 'Fa']},
    'HeatingQC': {'type': 'select', 'options': ['Ex', 'Gd', 'TA', 'Fa', 'Po']},
    'MasVnrType': {'type': 'select', 'options': ['BrkFace', 'None', 'Stone', 'BrkCmn']},
    'MasVnrArea': {'type': 'number'},
    'BsmtQual': {'type': 'select', 'options': ['Ex', 'Gd', 'TA', 'Fa', 'None']},
    'BsmtExposure': {'type': 'select', 'options': ['No', 'Gd', 'Mn', 'Av', 'None']},
    'FireplaceQu': {'type': 'select', 'options': ['Gd', 'TA', 'Ex', 'Po', 'Fa', 'None']},
    'PoolQC': {'type': 'select', 'options': ['Ex', 'Gd', 'TA', 'Fa', 'None']},
    'Fence': {'type': 'select', 'options': ['MnPrv', 'GdWo', 'GdPrv', 'MnWw', 'None']},
    'Alley': {'type': 'select', 'options': ['Grvl', 'Pave', 'None']},
    'MiscFeature': {'type': 'select', 'options': ['Shed', 'Gar2', 'Othr', 'TenC', 'None']},
}

@app.route('/')
def home():
    return render_template('index.html', input_features=INPUT_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    input_data_dict = {col: [np.nan] for col in final_columns if col != 'Id'}
    for key, value in form_data.items():
        expected_type = INPUT_FEATURES.get(key, {}).get('type')
        if expected_type == 'number':
            try:
                input_data_dict[key] = [float(value)]
            except:
                input_data_dict[key] = [0.0]
        else:
            input_data_dict[key] = [value]

    input_df = pd.DataFrame(input_data_dict)
    processed_input_df = input_df.copy()

    for col in categorical_features_for_onehot:
        if col not in processed_input_df.columns:
            processed_input_df[col] = 'None'
        processed_input_df[col] = processed_input_df[col].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        if col not in processed_input_df.columns:
            processed_input_df[col] = 0
        processed_input_df[col] = processed_input_df[col].fillna(0)

    processed_input_df['LotFrontage'] = processed_input_df.get('LotFrontage', 80.0)

    for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'OverallQual', 'OverallCond', 'GrLivArea']:
        if col in processed_input_df.columns:
            processed_input_df[col] = pd.to_numeric(processed_input_df[col], errors='coerce').fillna(0)

    processed_input_df['TotalSF'] = processed_input_df['TotalBsmtSF'] + processed_input_df['1stFlrSF'] + processed_input_df.get('2ndFlrSF', 0)
    processed_input_df['TotalBath'] = processed_input_df['FullBath'] + (0.5 * processed_input_df.get('HalfBath', 0)) + processed_input_df['BsmtFullBath'] + (0.5 * processed_input_df['BsmtHalfBath'])
    processed_input_df['YearBuiltRemod'] = processed_input_df['YearBuilt'] + processed_input_df['YearRemodAdd']
    processed_input_df['YrSold'] = pd.Timestamp.now().year
    processed_input_df['YearsSinceRemod'] = processed_input_df['YrSold'] - processed_input_df['YearRemodAdd']
    processed_input_df['YearsSinceBuilt'] = processed_input_df['YrSold'] - processed_input_df['YearBuilt']
    processed_input_df['HasPool'] = processed_input_df.get('PoolArea', 0).apply(lambda x: 1 if x > 0 else 0)
    processed_input_df['HasGarage'] = processed_input_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    processed_input_df['HasBsmt'] = processed_input_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    processed_input_df['HasFireplace'] = processed_input_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    processed_input_df['OverallQual_OverallCond'] = processed_input_df['OverallQual'] * processed_input_df.get('OverallCond', 5)
    processed_input_df['GrLivArea_OverallQual'] = processed_input_df['GrLivArea'] * processed_input_df['OverallQual']

    for col in quality_cols:
        processed_input_df[col] = processed_input_df.get(col, 'TA').map(quality_map).fillna(3)

    for feat in skewed_features_to_transform:
        if feat in processed_input_df.columns and processed_input_df[feat].dtype != object:
            processed_input_df[feat] = np.log1p(processed_input_df[feat])

    for col in categorical_features_for_onehot:
        if col in processed_input_df.columns:
            dummies = pd.get_dummies(processed_input_df[col], prefix=col, drop_first=True)
            processed_input_df = pd.concat([processed_input_df, dummies], axis=1)
            processed_input_df = processed_input_df.drop(columns=[col])

    final_input_df = pd.DataFrame(columns=final_columns)
    for col in final_columns:
        final_input_df[col] = processed_input_df.get(col, 0)

    if 'SalePrice' in final_input_df.columns:
        final_input_df = final_input_df.drop('SalePrice', axis=1)

    try:
        log_prediction = model.predict(final_input_df)[0]
        predicted_price = np.expm1(log_prediction)
        formatted_price = f"${predicted_price:,.2f}"
    except Exception as e:
        formatted_price = f"Error in prediction: {e}"

    return render_template('index.html', predicted_price=formatted_price, input_features=INPUT_FEATURES)

if __name__ == '__main__':
    app.run(debug=True)