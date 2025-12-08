import kagglehub
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataFrameProcessor:
    """
    A class designed to handle and process Pandas DataFrames (training and test sets),
    including methods for value counting, data type conversion, and outlier capping.
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Initializes the processor with training and test set DataFrames.

        Args:
            df_train (pd.DataFrame): The training dataset.
            df_test (pd.DataFrame): The test dataset.
        """
        self.df_train = df_train
        self.df_test = df_test

    def print_value_count(self, col: str):
        """
        Prints the unique values and their frequencies (value counts) for the given
        feature (column) in both the training and test sets, and displays the
        training set data type.

        Args:
            col (str): The name of the feature column to analyze.
        """
        print(f"--- Feature: {col} ---")
        print("\n[Train Data Value Counts]:")
        print(self.df_train[col].value_counts())

        print("\n[Test Data Value Counts]:")
        print(self.df_test[col].value_counts())

        print(f"\n[Train Data Type]: {self.df_train[col].dtype}")
        print("-" * (18 + len(col)))

    def replace_astype(self, col: str):
        """
        Removes underscores (_) from the specified feature column and converts
        its data type to integer (int).

        Note: This operation modifies self.df_train and self.df_test inplace.

        Args:
            col (str): The name of the feature column to process.
        """
        print(f"Processing column: {col}. Removing '_' and converting to integer...")

        # Remove underscores using regex
        self.df_train[col] = self.df_train[col].replace(r'\_', '', regex=True)
        self.df_test[col] = self.df_test[col].replace(r'\_', '', regex=True)


        # Convert to float first, then to integer for robust type conversion
        self.df_train[col] = self.df_train[col].astype(float).astype(int)
        self.df_test[col] = self.df_test[col].astype(float).astype(int)

        print(f"Column {col} successfully updated to type: {self.df_train[col].dtype}")

    def capping_quantile(self, col: str):
        """
        Performs outlier capping on the given feature column using the
        IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR). The bounds are calculated
        based *only* on the training set.

        Args:
            col (str): The name of the feature column to process.
        """
        print(f"\nApplying IQR Capping to column: {col}")

        # Calculate IQR bounds based on the training set
        Q1 = self.df_train[col].quantile(0.25)
        Q3 = self.df_train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        print(f"Calculated bounds (Lower, Upper): ({lower:.2f}, {upper:.2f})")

        # Apply clipping to both dataframes
        self.df_train[col] = self.df_train[col].clip(lower, upper)
        self.df_test[col] = self.df_test[col].clip(lower, upper)

        self.print_value_count(col)

    def capping_fixed(self, col: str, lower: float, upper: float):
        """
        Performs outlier capping on the given feature column using fixed,
        user-defined lower and upper bounds.

        Args:
            col (str): The name of the feature column to process.
            lower (float): The fixed lower bound for clipping.
            upper (float): The fixed upper bound for clipping.
        """
        print(f"\nApplying Fixed Capping to column: {col} with bounds ({lower}, {upper})")

        # Apply clipping to both dataframes
        self.df_train[col] = self.df_train[col].clip(lower, upper)
        self.df_test[col] = self.df_test[col].clip(lower, upper)

        self.print_value_count(col)

    def save_dataframes(self, base_filename: str):
        """
        Saves the processed df_train and df_test DataFrames to CSV files.

        The files will be saved as:
        1. [base_filename]_train.csv
        2. [base_filename]_test.csv

        Args:
            base_filename (str): The base name for the output files (e.g., 'processed_data').
        """
        train_path = f"{base_filename}_train.csv"
        test_path = f"{base_filename}_test.csv"

        # Save training data
        try:
            self.df_train.to_csv(train_path, index=False)
            print(f"Successfully saved training data to: {train_path}")
        except Exception as e:
            print(f"Error saving training data: {e}")

        # Save test data
        try:
            self.df_test.to_csv(test_path, index=False)
            print(f"Successfully saved test data to: {test_path}")
        except Exception as e:
            print(f"Error saving test data: {e}")

    def extract_loan_features(self, col: str = 'Type_of_Loan', loan_types: List[str] = None):
        """
        Extracts boolean features (one-hot encoding) from the specified loan type column
        for both df_train and df_test.

        New columns will be added to both dataframes inplace.

        Args:
            col (str): The column containing the concatenated loan types (default: 'Type_of_Loan').
            loan_types (List[str]): List of loan names to extract. If None, uses a default list.
        """

        if loan_types is None:
            # Default list based on the user's initial request
            loan_types = [
                'Credit-Builder Loan',
                'Personal Loan',
                'Student Loan',
                'Mortgage',
                'Home Equity Loan',
                'Payday Loan',
                'Other Loan',
                'Debt Consolidation'
            ]

        print(f"\nExtracting {len(loan_types)} boolean features from column: {col}...")

        # Process both DataFrames
        for df in [self.df_train, self.df_test]:
            for loan in loan_types:
                # Handle 'Debt Consolidation' matching 'Debt Consolidation Loan'
                search_string = loan if loan != 'Debt Consolidation' else 'Debt Consolidation Loan'

                # Create the new boolean column (True/False)
                # .str.contains() checks if the string contains the substring.
                # .fillna(False) sets NaN values (where the original cell was empty) to False.
                df[loan] = df[col].str.contains(search_string).fillna(False)

        print(f"Successfully added {len(loan_types)} new features to df_train and df_test.")
        # Optionally, print the first few rows of the new features
        print("\n[New Features Sample (Train Data)]")
        print(self.df_train[[*loan_types]].head())

    def impute_with_mean(self, col: str):
        """
        Performs mean imputation on the specified column for both df_train and df_test.
        The mean is calculated ONLY from the training data (df_train).

        Args:
            col (str): The name of the numerical column to impute.
        """
        print(f"\nImputing missing values in column '{col}' with mean of training data...")

        # 1. Calculate the mean ONLY from the training data
        train_mean = self.df_train[col].mean()

        print(f"Calculated training mean for '{col}': {train_mean:.2f}")

        # 2. Apply the calculated mean to fill NaNs in both DataFrames
        self.df_train[col] = self.df_train[col].fillna(train_mean)
        self.df_test[col] = self.df_test[col].fillna(train_mean)

        print(f"Imputation complete for '{col}'.")
        print(f"Remaining NaNs in df_train: {self.df_train[col].isnull().sum()}")
        print(f"Remaining NaNs in df_test: {self.df_test[col].isnull().sum()}")

    def impute_with_zero(self, col: str):
        """
        Performs zero imputation on the specified column for both df_train and df_test.

        Args:
            col (str): The name of the numerical column to impute.
        """
        print(f"\nImputing missing values in column '{col}' with 0...")
        self.df_train[col] = self.df_train[col].fillna(0)
        self.df_test[col] = self.df_test[col].fillna(0)

        print(f"Imputation complete for '{col}'.")
        print(f"Remaining NaNs in df_train: {self.df_train[col].isnull().sum()}")
        print(f"Remaining NaNs in df_test: {self.df_test[col].isnull().sum()}")

    def convert_credit_age(self, col: str = 'Credit_History_Age'):
        """
        Fills missing values and converts the credit history age column
        from "X Years and Y Months" string format to total months (integer).

        The placeholder "0 Years and 0 Months" is used for NaN values.

        Args:
            col (str): The name of the credit history age column.
        """

        placeholder = "0 Years and 0 Months"
        print(f"\nProcessing column: '{col}'. Filling NaNs and converting to total months...")

        # 1. Fill NaN values in both dataframes with the placeholder
        self.df_train[col] = self.df_train[col].fillna(placeholder)
        self.df_test[col] = self.df_test[col].fillna(placeholder)

        # Helper function to perform the extraction and calculation
        def calculate_total_months(df: pd.DataFrame, column_name: str) -> pd.Series:
            """Extracts years/months and calculates total months."""
            return (
                df[column_name]
                .str.extract(r'(\d+)\s+Years.*?(\d+)\s+Months')
                .astype(int)  # Convert extracted groups (years, months) to integers
                # Calculate total months: (Years * 12) + Months
                .apply(lambda x: x[0] * 12 + x[1], axis=1)
            )

        # 2. Apply the conversion to both dataframes
        self.df_train[col] = calculate_total_months(self.df_train, col)
        self.df_test[col] = calculate_total_months(self.df_test, col)

        print(f"✅ Conversion complete. '{col}' is now represented in total months (dtype: {self.df_train[col].dtype}).")

if __name__ == '__main__':
    # Reading and Exploding data
    print("Downloading the dataset")
    path = kagglehub.dataset_download("parisrohan/credit-score-classification")
    print("Path to dataset files:", path)

    print("Reading the dataset")
    df_train = pd.read_csv(path + "/train.csv")
    df_test = pd.read_csv(path + "/test.csv")

    print("Saving the original dataset")
    df_train.to_csv('original_train.csv')
    df_test.to_csv('original_test.csv')

    print("Exploring the data")
    print(df_train.info())
    print(df_test.info())
    print("Calculating the number of different values.")
    for col in df_test.columns:
        if col not in ['ID', 'Customer_ID', 'Name', 'SSN']:
            print(col, df_train[col].dtype, df_train[col].value_counts())
            print('-' * 100)
    print('*'*100)
    # Cleaning Data
    print("Cleaning the data")

    print("--- Initial Data Types---")
    print("Train Dtypes:\n", df_train.dtypes)
    print("\nTest Dtypes:\n", df_test.dtypes)
    print("-" * 40)

    # Create an instance of the processor, passing in the dataframes
    processor = DataFrameProcessor(df_train, df_test)

    for col in df_test.columns:
        print(f'Cleaning {col}')
        if col=='Age':
            processor.print_value_count('Age')
            processor.replace_astype('Age')
            processor.capping_fixed('Age', 0, 100)
        elif col=='Occupation':
            processor.df_train['Occupation'] = df_train['Occupation'].replace('_______', 'unknown_occupation')
            processor.df_test['Occupation'] = df_test['Occupation'].replace('_______', 'unknown_occupation')
            processor.print_value_count('Occupation')
        elif col=='Annual_Income':
            processor.print_value_count('Annual_Income')
            processor.replace_astype('Annual_Income')
            processor.capping_quantile('Annual_Income')
        elif col=='Monthly_Inhand_Salary':
            processor.print_value_count('Monthly_Inhand_Salary')
            processor.capping_quantile('Monthly_Inhand_Salary')
            processor.impute_with_mean('Monthly_Inhand_Salary')
        elif col=='Num_Bank_Accounts':
            processor.print_value_count('Num_Bank_Accounts')
            processor.capping_quantile('Num_Bank_Accounts')
            processor.capping_fixed('Num_Bank_Accounts', 0, 20)
        elif col=='Num_Credit_Card':
            processor.print_value_count('Num_Credit_Card')
            processor.capping_quantile('Num_Credit_Card')
            processor.replace_astype('Num_Credit_Card')
        elif col=='Interest_Rate':
            processor.print_value_count('Interest_Rate')
            processor.capping_quantile('Interest_Rate')
        elif col=='Num_of_Loan':
            processor.print_value_count('Num_of_Loan')
            processor.replace_astype('Num_of_Loan')
            processor.capping_quantile('Num_of_Loan')
            processor.capping_fixed('Num_of_Loan', 0, 11)
        elif col=='Type_of_Loan':
            processor.print_value_count('Type_of_Loan')
            processor.extract_loan_features('Type_of_Loan')
        elif col=='Delay_from_due_date':
            processor.print_value_count('Delay_from_due_date')
            processor.capping_quantile('Delay_from_due_date')
            processor.capping_fixed('Delay_from_due_date', 0, 100)
        elif col=='Num_of_Delayed_Payment':
            processor.print_value_count('Num_of_Delayed_Payment')
            processor.impute_with_zero('Num_of_Delayed_Payment')
            processor.replace_astype('Num_of_Delayed_Payment')
            processor.capping_quantile('Num_of_Delayed_Payment')
            processor.capping_fixed('Num_of_Delayed_Payment', 0, 100)
        elif col=='Changed_Credit_Limit':
            processor.df_train['Changed_Credit_Limit'] = processor.df_train['Changed_Credit_Limit'].replace(r'[_\-]', '', regex=True)
            processor.df_test['Changed_Credit_Limit'] = processor.df_test['Changed_Credit_Limit'].replace(r'[_\-]', '', regex=True)
            processor.df_train['Changed_Credit_Limit'] = pd.to_numeric(df_train['Changed_Credit_Limit'], errors='coerce')
            processor.df_test['Changed_Credit_Limit'] = pd.to_numeric(df_test['Changed_Credit_Limit'], errors='coerce')
            processor.impute_with_mean('Changed_Credit_Limit')
        elif col=='Num_Credit_Inquiries':
            processor.print_value_count('Num_Credit_Inquiries')
            processor.impute_with_mean('Num_Credit_Inquiries')
        elif col=='Credit_Mix':
            processor.print_value_count('Credit_Mix')
            processor.df_train['Credit_Mix'] = processor.df_train['Credit_Mix'].replace('_', 'unknown_credit_mix')
            processor.df_test['Credit_Mix'] = processor.df_test['Credit_Mix'].replace('_', 'unknown_credit_mix')
        elif col=='Outstanding_Debt':
            processor.print_value_count('Outstanding_Debt')
            processor.df_train['Outstanding_Debt'] = processor.df_train['Outstanding_Debt'].replace(r'[_\-]', '', regex=True)
            processor.df_test['Outstanding_Debt'] = processor.df_test['Outstanding_Debt'].replace(r'[_\-]', '', regex=True)
            df_train['Outstanding_Debt'] = pd.to_numeric(df_train['Outstanding_Debt'], errors='coerce')
            df_test['Outstanding_Debt'] = pd.to_numeric(df_test['Outstanding_Debt'], errors='coerce')
        elif col=='Credit_Utilization_Ratio':
            processor.print_value_count('Credit_Utilization_Ratio')
            processor.replace_astype('Credit_Utilization_Ratio')
        elif col=='Credit_History_Age':
            processor.print_value_count('Credit_History_Age')
            processor.convert_credit_age('Credit_History_Age')
        elif col=='Payment_of_Min_Amount':
            processor.print_value_count('Payment_of_Min_Amount')
            processor.df_train['Payment_of_Min_Amount'] = processor.df_train['Payment_of_Min_Amount'].replace('NM', 'No')
            processor.df_test['Payment_of_Min_Amount'] = processor.df_test['Payment_of_Min_Amount'].replace('NM', 'No')
        elif col=='Amount_invested_monthly':
            processor.print_value_count('Amount_invested_monthly')
            processor.impute_with_zero('Amount_invested_monthly')
            processor.replace_astype('Amount_invested_monthly')
        elif col=='Payment_Behaviour':
            processor.print_value_count('Payment_Behaviour')
            processor.df_train['Payment_Behaviour'] = processor.df_train['Payment_Behaviour'].replace('!@9#%8', 'unknown_payment_behaviour')
            processor.df_test['Payment_Behaviour'] = processor.df_test['Payment_Behaviour'].replace('!@9#%8', 'unknown_payment_behaviour')
        elif col=='Monthly_Balance':
            processor.print_value_count('Monthly_Balance')
            processor.impute_with_zero('Monthly_Balance')
            processor.replace_astype('Monthly_Balance')
        print(f'Sucessfully cleaned {col}')

    print("\n--- Final DataFrames  ---")
    print("Modified Train Data (df_train):")
    print(processor.df_train.head())
    print("\nModified Test Data (df_test):")
    print(processor.df_test.head())

    #Saving the cleaned Data
    processor.save_dataframes('cleaned_data')


