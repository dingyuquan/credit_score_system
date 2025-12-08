import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from typing import Tuple

class FeatureEncoder:
    """
    A class to handle feature encoding and scaling for machine learning datasets.
    It synchronously processes training (df_train) and testing (df_test) data
    while preventing data leakage.
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Initializes the encoder with training and test set DataFrames.

        Args:
            df_train (pd.DataFrame): The training dataset.
            df_test (pd.DataFrame): The test dataset.
        """
        # Store original dataframes
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()

        # Initialize transformers
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()

        # Placeholders for encoded data
        self.X_train_final = None
        self.X_test_final = None
        self.y_train_encoded = None

        # Define columns to drop (as specified in your code)
        self.id_cols = ['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan']
        self.target_col = 'Credit_Score'

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handles feature and target separation and column dropping."""
        print("--- 1. Data Separation and Dropping ---")

        # Drop identifier and target columns from X sets
        X_train = self.df_train.drop(columns=[*self.id_cols, self.target_col], errors='ignore')
        X_test = self.df_test.drop(columns=self.id_cols, errors='ignore')

        # Extract the target variable (Credit_Score) from the training data
        y_train = self.df_train[self.target_col]

        # Label Encoding for the Target Variable (y)
        self.y_train_encoded = self.label_encoder.fit_transform(y_train)
        print(f"Target variable ('{self.target_col}') encoded. Unique classes: {self.label_encoder.classes_}")

        return X_train, X_test

    def numerical_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardizes numerical features using StandardScaler."""
        print("\n--- 2. Numerical Standardization (StandardScaler) ---")

        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        print(f"Numerical columns found: {list(numerical_cols)}")

        # Fit the scaler ONLY on the training data
        X_train_num_scaled = self.scaler.fit_transform(X_train[numerical_cols])
        # Transform both training and test data
        X_test_num_scaled = self.scaler.transform(X_test[numerical_cols])

        # Convert back to DataFrame, preserving column names and indices
        X_train_num_encoded = pd.DataFrame(X_train_num_scaled, columns=numerical_cols, index=X_train.index)
        X_test_num_encoded = pd.DataFrame(X_test_num_scaled, columns=numerical_cols, index=X_test.index)

        return X_train_num_encoded, X_test_num_encoded

    def onehot_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs One-Hot Encoding on categorical (object/bool) features."""
        print("\n--- 3. Categorical Encoding (OneHotEncoder) ---")

        cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns
        print(f"Categorical columns found: {list(cat_cols)}")

        # Fit the encoder ONLY on the training data
        self.onehot_encoder.fit(X_train[cat_cols])

        # Transform both training and test data
        X_train_cat_encoded_array = self.onehot_encoder.transform(X_train[cat_cols])
        X_test_cat_encoded_array = self.onehot_encoder.transform(X_test[cat_cols])

        # Convert back to DataFrame, preserving column names and indices
        feature_names = self.onehot_encoder.get_feature_names_out(cat_cols)

        X_train_cat_encoded = pd.DataFrame(X_train_cat_encoded_array, columns=feature_names, index=X_train.index)
        X_test_cat_encoded = pd.DataFrame(X_test_cat_encoded_array, columns=feature_names, index=X_test.index)

        return X_train_cat_encoded, X_test_cat_encoded

    def run_encoding(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Executes the entire feature encoding and scaling pipeline.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
                (X_train_final, X_test_final, y_train_encoded)
        """
        # Step 1: Separate and drop columns
        X_train, X_test = self.prepare_data()

        # Step 2: Numerical standardization
        X_train_num_encoded, X_test_num_encoded = self.numerical_encoding(X_train.copy(), X_test.copy())

        # Step 3: Categorical One-Hot Encoding
        X_train_cat_encoded, X_test_cat_encoded = self.onehot_encoding(X_train.copy(), X_test.copy())

        # Step 4: Concatenate numerical and categorical features
        print("\n--- 4. Concatenating Features ---")

        X_train_final_tmp = pd.concat([X_train_num_encoded, X_train_cat_encoded], axis=1)
        X_test_final_tmp = pd.concat([X_test_num_encoded, X_test_cat_encoded], axis=1)

        y_series = pd.Series(self.y_train_encoded, name='target_encoded', index=X_train_final_tmp.index)


        self.X_train_final = pd.concat([X_train_final_tmp, y_series], axis=1)


        self.X_test_final = X_test_final_tmp

        print(f"Final X_train shape: {self.X_train_final.shape}")
        print(f"Final X_test shape: {self.X_test_final.shape}")

        return self.X_train_final, self.X_test_final





if __name__ == "__main__":
    df_train = pd.read_csv('cleaned_data_train.csv')
    df_test = pd.read_csv('cleaned_data_test.csv')
    encoder = FeatureEncoder(df_train, df_test)


    X_train_final, X_test_final = encoder.run_encoding()


    print("\n--- Final Encoded Data Samples ---")
    print("X_train_final (first 5 rows):")
    print(X_train_final.head())

    base_filename='encoded_data'
    train_path = f"{base_filename}_train.csv"
    test_path = f"{base_filename}_test.csv"

    print(f"\n--- 5. Saving Final DataFrames ---")

    # Save final training data (with encoded target)
    try:
        X_train_final.to_csv(train_path, index=False)
        print(f"Successfully saved final training data to: {train_path}")
    except Exception as e:
        print(f"Error saving training data: {e}")

    # Save final test data
    try:
        X_test_final.to_csv(test_path, index=False)
        print(f"Successfully saved final test data to: {test_path}")
    except Exception as e:
        print(f"Error saving test data: {e}")