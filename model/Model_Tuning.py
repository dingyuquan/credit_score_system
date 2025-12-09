# Hyperparameter tuning script for XGBoost using Optuna.

import numpy as np
import pandas as pd
import optuna
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


# Optimal feature set obtained from previous XGBoost feature selection experiments
optimal_features = [
    "Credit_Mix_Good",
    "Credit_Mix_Standard",
    "Payment_of_Min_Amount_No",
    "Credit_Mix_unknown_credit_mix",
    "Outstanding_Debt",
    "Credit_Mix_Bad",
    "Interest_Rate",
    "Month_February",
    "Month_January",
    "Num_Credit_Card",
    "Month_March",
    "Delay_from_due_date",
    "Num_Bank_Accounts",
    "Changed_Credit_Limit",
    "Payment_Behaviour_Low_spent_Small_value_payments",
    "Total_EMI_per_month",
    "Num_Credit_Inquiries",
    "Month_July",
    "Credit-Builder Loan_False",
    "Annual_Income",
    "Num_of_Loan",
    "Occupation_Entrepreneur",
    "Num_of_Delayed_Payment",
    "Month_August",
    "Occupation_Engineer",
    "Personal Loan_False",
    "Debt Consolidation_False",
    "Credit_History_Age",
    "Occupation_Writer",
]


class XGBOptunaTuner:
    """
    A class that encapsulates XGBoost training and hyperparameter optimization using Optuna.

    - Subsets data to optimal_features
    - Performs Stratified K-Fold cross-validation inside Optuna
    - Searches best hyperparameters
    - Trains the best model
    - Evaluates model on validation set
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_train_final,
        X_test_final,
        optimal_features,
        random_state: int = 42,
    ):
        self.optimal_features = optimal_features

        # Select only the optimal features
        self.X_train_opt = X_train[optimal_features].copy()
        self.X_val_opt = X_val[optimal_features].copy()
        self.X_full_opt = X_train_final[optimal_features].copy()
        self.X_test_opt = X_test_final[optimal_features].copy()

        self.y_train = np.array(y_train)
        self.y_val = np.array(y_val)

        self.num_classes = len(np.unique(self.y_train))
        self.random_state = random_state

        self.study = None
        self.best_model = None

        print("X_train_opt shape:", self.X_train_opt.shape)
        print("X_val_opt shape:", self.X_val_opt.shape)

    def _base_params(self) -> dict:
        """Returns fixed XGBoost parameters used across all Optuna trials."""
        return {
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function used by Optuna to evaluate each hyperparameter set."""

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        }
        params.update(self._base_params())

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = []

        # Cross-validation loop
        for train_idx, valid_idx in skf.split(self.X_train_opt, self.y_train):
            X_tr = self.X_train_opt.iloc[train_idx]
            X_va = self.X_train_opt.iloc[valid_idx]
            y_tr = self.y_train[train_idx]
            y_va = self.y_train[valid_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            y_proba = model.predict_proba(X_va)
            auc = roc_auc_score(y_va, y_proba, multi_class="ovr")
            scores.append(auc)

        return float(np.mean(scores))

    def run_optimization(self, n_trials: int = 50, study_name: str = "xgb_optimal_features"):
        """Runs Optuna hyperparameter search."""
        self.study = optuna.create_study(direction="maximize", study_name=study_name)
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        print("\nCompleted trials:", len(self.study.trials))
        print("Best trial AUC:", self.study.best_value)
        print("Best parameters:")
        for k, v in self.study.best_trial.params.items():
            print(f"  {k}: {v}")

    def train_best_model(self):
        """Trains XGBoost using the best hyperparameters found by Optuna."""
        if self.study is None:
            raise RuntimeError("Optimization has not been run yet.")

        best_params = self.study.best_trial.params.copy()
        best_params.update(self._base_params())

        self.best_model = XGBClassifier(**best_params)
        self.best_model.fit(self.X_train_opt, self.y_train)
        return self.best_model

    def evaluate_on_validation(self) -> float:
        """Evaluates trained model on validation set and prints performance metrics."""
        if self.best_model is None:
            raise RuntimeError("No trained model found. Run train_best_model() first.")

        y_pred = self.best_model.predict(self.X_val_opt)
        y_proba = self.best_model.predict_proba(self.X_val_opt)

        print("\nValidation Classification Report:")
        print(classification_report(self.y_val, y_pred))

        print("\nValidation Confusion Matrix:")
        print(confusion_matrix(self.y_val, y_pred))

        auc = roc_auc_score(self.y_val, y_proba, multi_class="ovr")
        print("\nValidation macro ROC-AUC: {:.4f}".format(auc))

        return auc


if __name__ == "__main__":
    df_train_enc = pd.read_csv("encoded_data_train.csv")
    df_test_enc = pd.read_csv("encoded_data_test.csv")

    # Target column created during encoding
    y_full = df_train_enc["target_encoded"]
    X_full = df_train_enc.drop(columns=["target_encoded"])

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )

    # Full training and test sets (for later final training or prediction)
    X_train_final = X_full.copy()
    X_test_final = df_test_enc.copy()

    # Initialize tuner
    tuner = XGBOptunaTuner(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_train_final=X_train_final,
        X_test_final=X_test_final,
        optimal_features=optimal_features,
        random_state=42,
    )

# Run optimization
tuner.run_optimization(n_trials=50)

# Train best model
best_model = tuner.train_best_model()

# Save the best model
joblib.dump(best_model, "xgb_best_model.joblib")
print("Best XGBoost model saved as xgb_best_model.joblib")

# Evaluate on validation set
tuner.evaluate_on_validation()

