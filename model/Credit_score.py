import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost
import joblib
import os


def Credit_Score(user_info, df_train, model):
    """
    The function can predict the credit score of a user.
    """

    """
    INPUT:
    user_info : It should be a dict with right key.
    {
      Age: int,
      Gender: str,
      SSN: str,
      Legal_name: str,
      Annual_Income: int,
      Monthly_Inhand_Salary: int,
      Occupation: float
      Month: str (January... December)[Application Month] use time library to get the current month
    }
    df_train : Training dataset
    model :The best model, it's a joblib file.
  
    OUTPUT:
    The function will return the credit score of the user.
    """
    df_user_info = pd.DataFrame(user_info, index=[0])
    X_train = df_train.drop(['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Credit_Score'], axis=1)

    target_keys = ['Month', 'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary']

    filtered_dict = {
        key: user_info[key]
        for key in target_keys
        if key in user_info
    }
    df_user_info = pd.DataFrame(filtered_dict, index=[0])

    for col in X_train.columns:
        if col not in df_user_info.columns:
            df_user_info[col] = X_train[col].mode()
    print(df_user_info)

    # Transform input categorical data
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns
    # print(cat_cols)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_train[cat_cols])

    df_user_info_cat = df_user_info[cat_cols]
    encoded_data_array = encoder.transform(df_user_info_cat)

    new_col_names = encoder.get_feature_names_out(cat_cols)

    encoded_df_user_info_cat = pd.DataFrame(encoded_data_array, columns=new_col_names)
    print(encoded_df_user_info_cat)

    # Transform input numerical data
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    # print(num_cols)
    scaler = StandardScaler()

    scaler.fit(X_train[num_cols])
    X_train_num_encoded = scaler.transform(df_user_info[num_cols])
    print(X_train_num_encoded)
    X_train_num_df = pd.DataFrame(
        X_train_num_encoded,
        columns=num_cols,
        index=df_user_info.index
    )

    print("\n--- 转换后的 DataFrame (包含列名和索引) ---")
    # print(X_train_num_df)
    user_info_final = pd.concat([X_train_num_df, encoded_df_user_info_cat], axis=1)
    print(user_info_final)

    optimal_features = [
        'Credit_Mix_Good',
        'Credit_Mix_Standard',
        'Payment_of_Min_Amount_No',
        'Credit_Mix_unknown_credit_mix',
        'Outstanding_Debt',
        'Credit_Mix_Bad',
        'Interest_Rate',
        'Month_February',
        'Month_January',
        'Num_Credit_Card',
        'Month_March',
        'Delay_from_due_date',
        'Num_Bank_Accounts',
        'Changed_Credit_Limit',
        'Payment_Behaviour_Low_spent_Small_value_payments',
        'Total_EMI_per_month',
        'Num_Credit_Inquiries',
        'Month_July',
        'Credit-Builder Loan_False',
        'Annual_Income',
        'Num_of_Loan',
        'Occupation_Entrepreneur',
        'Num_of_Delayed_Payment',
        'Month_August',
        'Occupation_Engineer',
        'Personal Loan_False',
        'Debt Consolidation_False',
        'Credit_History_Age',
        'Occupation_Writer'
    ]
    return model.predict(user_info_final[optimal_features])


if __name__ == "__main__":
    user_info = {
        "Age": 20,
        "Gender": "Male",
        "SSN": "100-100-100",
        "Legal_name": "Bobby Han",
        "Annual_Income": 100000,
        "Monthly_Inhand_Salary": 10000,
        "Occupation": "Scientist",
        "Month": "December"
    }
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'cleaned_train_data.csv')
    model_path = os.path.join(script_dir, 'best_model.joblib')
    df_loaded = pd.read_csv(csv_path)
    model = joblib.load(model_path)

    print(df_loaded.head())
    print(model)
    print(Credit_Score(user_info, df_loaded, model))


