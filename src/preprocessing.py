from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    working_train_df = working_train_df.reset_index().drop("index",axis=1)
    working_val_df = working_val_df.reset_index().drop("index",axis=1)
    working_test_df = working_test_df.reset_index().drop("index",axis=1)

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    cols_numeric = [col for col in working_train_df.columns if working_train_df[col].dtype != 'object']
    cols_object = [col for col in working_train_df.columns if working_train_df[col].dtype == 'object']
    cols_ordinal = [col for col in cols_object if working_train_df[col].nunique() == 2]
    cols_oneHot = [col for col in cols_object if working_train_df[col].nunique() > 2]

    encoder_ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    encoder_oneHot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    encoded_ordinal = encoder_ordinal.fit(working_train_df[cols_ordinal])
    encoded_oneHot = encoder_oneHot.fit(working_train_df[cols_oneHot])
    columns_oneHot_encoded = encoded_oneHot.get_feature_names_out(cols_oneHot)

    working_train_df_cols_ord = encoded_ordinal.transform(working_train_df[cols_ordinal])
    working_train_df_cols_oneH = encoded_oneHot.transform(working_train_df[cols_oneHot])

    working_val_df_cols_ord = encoded_ordinal.transform(working_val_df[cols_ordinal])
    working_val_df_cols_oneH = encoded_oneHot.transform(working_val_df[cols_oneHot])

    working_test_df_cols_ord = encoded_ordinal.transform(working_test_df[cols_ordinal])
    working_test_df_cols_oneH = encoded_oneHot.transform(working_test_df[cols_oneHot])
    
    #Use append for union of df

    working_train_df_encoded = pd.concat([
        working_train_df[cols_numeric],
        pd.DataFrame(working_train_df_cols_ord, columns = cols_ordinal ), 
        pd.DataFrame(working_train_df_cols_oneH, columns = columns_oneHot_encoded)]
        , axis = 1)

    working_val_df_encoded = pd.concat([
        working_val_df[cols_numeric],
        pd.DataFrame(working_val_df_cols_ord, columns = cols_ordinal ), 
        pd.DataFrame(working_val_df_cols_oneH, columns = columns_oneHot_encoded)]
        , axis = 1)

    working_test_df_encoded = pd.concat([
        working_test_df[cols_numeric],
        pd.DataFrame(working_test_df_cols_ord, columns = cols_ordinal ), 
        pd.DataFrame(working_test_df_cols_oneH, columns = columns_oneHot_encoded)]
        , axis = 1)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.
    imputer = SimpleImputer(strategy = "mean")
    imputer_mean = imputer.fit(working_train_df_encoded)

    working_train_df_imputed = imputer_mean.transform(working_train_df_encoded)
    working_val_df_imputed = imputer_mean.transform(working_val_df_encoded)
    working_test_df_imputed = imputer_mean.transform(working_test_df_encoded)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    scaler = MinMaxScaler()
    scaler_minmax = scaler.fit(working_train_df_imputed)

    working_train_df_scaled = scaler_minmax.transform(working_train_df_imputed)
    working_val_df_scaled = scaler_minmax.transform(working_val_df_imputed)
    working_test_df_scaled = scaler_minmax.transform(working_test_df_imputed)

    return working_train_df_scaled, working_val_df_scaled, working_test_df_scaled
