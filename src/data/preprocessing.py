import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_credit_data(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    Preprocess dataset:
    - Rename columns
    - Clean EDUCATION and MARRIAGE categories
    - Merge PAY_i categories
    - Drop ID
    - One-hot encode categorical features
    - Save to CSV if save_path is provided

    Parameters:
        df (pd.DataFrame): Raw dataframe
        save_path (str, optional): Path to save the processed CSV

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()

    # Rename columns
    df = df.rename(columns={'default payment next month': 'default_pnm', 'PAY_0': 'PAY_1'})

    # EDUCATION: group rare/unknown categories
    idx_ed = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
    df.loc[idx_ed, 'EDUCATION'] = 4

    # MARRIAGE: group 0 (?) into 3
    idx_marr = df.MARRIAGE == 0
    df.loc[idx_marr, 'MARRIAGE'] = 3

    # PAY_1 to PAY_6: merge -2 and 0 into -1
    for i in range(1, 7):
        idx = (df[f'PAY_{i}'] == -2) | (df[f'PAY_{i}'] == 0)
        df.loc[idx, f'PAY_{i}'] = -1

    # Drop ID
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # One-hot encoding for categorical variables
    features_ohe = ['SEX', 'MARRIAGE', 'EDUCATION']
    df_categorical = df[features_ohe]
    ohe = OneHotEncoder(sparse_output=False)
    encoded_features = ohe.fit_transform(df_categorical)
    df_ohe = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(features_ohe))
    df_encoded = pd.concat([df.drop(columns=features_ohe), df_ohe], axis=1)

    # Save if path provided
    if save_path:
        df_encoded.to_csv(save_path, index=False)
        print(f"Preprocessed dataset saved to {save_path}")

    return df_encoded
