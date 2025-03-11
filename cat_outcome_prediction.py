import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_additional_features(df):
    if 'outcome_weekday' in df.columns:
        df['is_weekend'] = df['outcome_weekday'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    if 'outcome_month' in df.columns:
        df['is_kitten_season'] = df['outcome_month'].apply(lambda x: 1 if x in [3, 4, 5, 6] else 0)

    if 'outcome_hour' in df.columns:
        df['during_business_hours'] = df['outcome_hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)

    if 'outcome_age_(days)' in df.columns:
        def age_to_months_category(days):
            months = days / 30.5
            if months < 2:
                return "0-2months"
            elif months < 6:
                return "2-6months"
            elif months < 12:
                return "6-12months"
            elif months < 36:
                return "1-3years"
            elif months < 84:
                return "3-7years"
            else:
                return "7+years"

        df['age_months_category'] = df['outcome_age_(days)'].apply(age_to_months_category)

    return df

def apply_smote_balancing(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nClass distribution before SMOTE:")
    original_counts = np.bincount(y_train)
    for i, count in enumerate(original_counts):
        print(f"  Class {i}: {count} ({count/len(y_train)*100:.2f}%)")

    print("\nClass distribution after SMOTE:")
    balanced_counts = np.bincount(y_train_resampled)
    for i, count in enumerate(balanced_counts):
        print(f"  Class {i}: {count} ({count/len(y_train_resampled)*100:.2f}%)")

    return X_train_resampled, y_train_resampled

def getUnfitModels():
    models = list()
    models.append(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED))
    models.append(LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=RANDOM_SEED))
    models.append(LogisticRegression(max_iter=1000, C=10, class_weight='balanced', random_state=RANDOM_SEED))
    return models

def evaluateModel(y_true, predictions, model_name):
    accuracy = accuracy_score(y_true, predictions)
    f1_macro = f1_score(y_true, predictions, average='macro')
    precision = precision_score(y_true, predictions, average='macro')
    recall = recall_score(y_true, predictions, average='macro')

    print(f"\n*** {model_name} Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }

def fitBaseModels(X_train, y_train, X_val, models):
    dfPredictions = pd.DataFrame()

    for i, model in enumerate(models):
        model_name = f"LogisticRegression_{i+1}"
        print(f"Training base model {i+1}/{len(models)}: {model_name}")

        model.fit(X_train, y_train)

        predictions = model.predict_proba(X_val)

        for j in range(predictions.shape[1]):
            colName = f"model_{i}_class_{j}"
            dfPredictions[colName] = predictions[:, j]

    return dfPredictions, models

def fitStackedModel(X, y):
    print("Training stacked model (meta-learner)...")
    meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    meta_learner.fit(X, y)
    return meta_learner

def create_bagged_model(X_train, y_train):
    print("\n===== Training Bagging Classifier =====")
    base_estimator = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)

    bagging = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=RANDOM_SEED
    )

    print(f"Training BaggingClassifier with LogisticRegression...")
    bagging.fit(X_train, y_train)
    return bagging

def create_voting_classifier(X_train, y_train):
    print("\n===== Training Voting Classifier =====")

    lr1 = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=RANDOM_SEED)
    lr2 = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=RANDOM_SEED)
    lr3 = LogisticRegression(max_iter=1000, C=10.0, class_weight='balanced', random_state=RANDOM_SEED)

    voting_clf = VotingClassifier(
        estimators=[
            ('lr_small_c', lr1),
            ('lr_default_c', lr2),
            ('lr_large_c', lr3)
        ],
        voting='soft'
    )

    print("Training VotingClassifier...")
    voting_clf.fit(X_train, y_train)
    return voting_clf

def perform_kfold_cross_validation(X_data, y_data, label_encoder, n_splits=5):
    print("\n===== Cross-Validation with Required KFold Format =====")
    print(f"X data shape: {X_data.shape}")

    model_names = ['LogisticRegression', 'BaggingClassifier', 'VotingClassifier', 'StackedModel']

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_indices = list(kfold.split(X_data))

    fold_results = {name: {'accuracy': [], 'f1_macro': [], 'precision': [], 'recall': []}
                    for name in model_names}

    print(f"\n===== Evaluating LogisticRegression across {n_splits} folds =====")
    for fold, (train_index, test_index) in enumerate(fold_indices, 1):
        print(f"\n--- LogisticRegression: Fold {fold}/{n_splits} ---")

        X_fold_train, X_fold_test = X_data[train_index], X_data[test_index]
        y_fold_train, y_fold_test = y_data[train_index], y_data[test_index]

        X_fold_train_resampled, y_fold_train_resampled = apply_smote_balancing(X_fold_train, y_fold_train)

        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
        lr_model.fit(X_fold_train_resampled, y_fold_train_resampled)

        y_pred = lr_model.predict(X_fold_test)

        y_pred_original = label_encoder.inverse_transform(y_pred)
        y_test_original = label_encoder.inverse_transform(y_fold_test)

        metrics = evaluateModel(y_test_original, y_pred_original, "LogisticRegression")

        fold_results['LogisticRegression']['accuracy'].append(metrics['accuracy'])
        fold_results['LogisticRegression']['f1_macro'].append(metrics['f1_macro'])
        fold_results['LogisticRegression']['precision'].append(metrics['precision'])
        fold_results['LogisticRegression']['recall'].append(metrics['recall'])

    print(f"\n===== Evaluating BaggingClassifier across {n_splits} folds =====")
    for fold, (train_index, test_index) in enumerate(fold_indices, 1):
        print(f"\n--- BaggingClassifier: Fold {fold}/{n_splits} ---")

        X_fold_train, X_fold_test = X_data[train_index], X_data[test_index]
        y_fold_train, y_fold_test = y_data[train_index], y_data[test_index]

        X_fold_train_resampled, y_fold_train_resampled = apply_smote_balancing(X_fold_train, y_fold_train)

        bagging_clf = create_bagged_model(X_fold_train_resampled, y_fold_train_resampled)
        y_bagging_pred = bagging_clf.predict(X_fold_test)
        y_bagging_pred_original = label_encoder.inverse_transform(y_bagging_pred)
        y_test_original = label_encoder.inverse_transform(y_fold_test)

        bagging_metrics = evaluateModel(y_test_original, y_bagging_pred_original, "BaggingClassifier")
        fold_results['BaggingClassifier']['accuracy'].append(bagging_metrics['accuracy'])
        fold_results['BaggingClassifier']['f1_macro'].append(bagging_metrics['f1_macro'])
        fold_results['BaggingClassifier']['precision'].append(bagging_metrics['precision'])
        fold_results['BaggingClassifier']['recall'].append(bagging_metrics['recall'])

    print(f"\n===== Evaluating VotingClassifier across {n_splits} folds =====")
    for fold, (train_index, test_index) in enumerate(fold_indices, 1):
        print(f"\n--- VotingClassifier: Fold {fold}/{n_splits} ---")

        X_fold_train, X_fold_test = X_data[train_index], X_data[test_index]
        y_fold_train, y_fold_test = y_data[train_index], y_data[test_index]

        X_fold_train_resampled, y_fold_train_resampled = apply_smote_balancing(X_fold_train, y_fold_train)

        voting_clf = create_voting_classifier(X_fold_train_resampled, y_fold_train_resampled)
        y_voting_pred = voting_clf.predict(X_fold_test)
        y_voting_pred_original = label_encoder.inverse_transform(y_voting_pred)
        y_test_original = label_encoder.inverse_transform(y_fold_test)

        voting_metrics = evaluateModel(y_test_original, y_voting_pred_original, "VotingClassifier")
        fold_results['VotingClassifier']['accuracy'].append(voting_metrics['accuracy'])
        fold_results['VotingClassifier']['f1_macro'].append(voting_metrics['f1_macro'])
        fold_results['VotingClassifier']['precision'].append(voting_metrics['precision'])
        fold_results['VotingClassifier']['recall'].append(voting_metrics['recall'])

    print(f"\n===== Evaluating StackedModel across {n_splits} folds =====")
    for fold, (train_index, test_index) in enumerate(fold_indices, 1):
        print(f"\n--- StackedModel: Fold {fold}/{n_splits} ---")

        X_fold_train, X_fold_test = X_data[train_index], X_data[test_index]
        y_fold_train, y_fold_test = y_data[train_index], y_data[test_index]

        X_fold_train_resampled, y_fold_train_resampled = apply_smote_balancing(X_fold_train, y_fold_train)

        model_list = getUnfitModels()
        dfPredictions, fitted_models = fitBaseModels(X_fold_train_resampled, y_fold_train_resampled, X_fold_train_resampled, model_list.copy())
        stacked_model = fitStackedModel(dfPredictions, y_fold_train_resampled)

        dfTestPredictions = pd.DataFrame()
        for i, model in enumerate(fitted_models):
            test_predictions = model.predict_proba(X_fold_test)
            for j in range(test_predictions.shape[1]):
                colName = f"model_{i}_class_{j}"
                dfTestPredictions[colName] = test_predictions[:, j]

        y_stacked_pred = stacked_model.predict(dfTestPredictions)
        y_stacked_pred_original = label_encoder.inverse_transform(y_stacked_pred)
        y_test_original = label_encoder.inverse_transform(y_fold_test)

        stacked_metrics = evaluateModel(y_test_original, y_stacked_pred_original, "StackedModel")
        fold_results['StackedModel']['accuracy'].append(stacked_metrics['accuracy'])
        fold_results['StackedModel']['f1_macro'].append(stacked_metrics['f1_macro'])
        fold_results['StackedModel']['precision'].append(stacked_metrics['precision'])
        fold_results['StackedModel']['recall'].append(stacked_metrics['recall'])

    print("\n===== Cross-Validation Results =====")
    summary_data = []

    for model_name, metrics in fold_results.items():
        summary = {
            'model_name': model_name,
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'f1_macro_mean': np.mean(metrics['f1_macro']),
            'f1_macro_std': np.std(metrics['f1_macro']),
            'precision_mean': np.mean(metrics['precision']),
            'precision_std': np.std(metrics['precision']),
            'recall_mean': np.mean(metrics['recall']),
            'recall_std': np.std(metrics['recall'])
        }
        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('f1_macro_mean', ascending=False)

    print("\nModel performance summary (averaged across all folds):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_df.to_string(index=False))

    best_model = summary_df.iloc[0]['model_name']
    print(f"\nBest model based on F1 score: {best_model}")
    print(f"  Average F1 Score: {summary_df.iloc[0]['f1_macro_mean']:.4f} (±{summary_df.iloc[0]['f1_macro_std']:.4f})")

    return summary_df

def train_and_evaluate_models_on_test(X_train, y_train, X_test, y_test, label_encoder):
    print("\n===== Final Model Training and Evaluation on Test Set =====")

    X_train_resampled, y_train_resampled = apply_smote_balancing(X_train, y_train)

    test_results = []

    print("\nTraining LogisticRegression...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    lr_model.fit(X_train_resampled, y_train_resampled)

    y_lr_pred = lr_model.predict(X_test)
    y_lr_pred_original = label_encoder.inverse_transform(y_lr_pred)
    y_test_original = label_encoder.inverse_transform(y_test)

    lr_metrics = evaluateModel(y_test_original, y_lr_pred_original, "LogisticRegression")
    test_results.append(lr_metrics)

    bagging_clf = create_bagged_model(X_train_resampled, y_train_resampled)
    y_bagging_pred = bagging_clf.predict(X_test)
    y_bagging_pred_original = label_encoder.inverse_transform(y_bagging_pred)

    bagging_metrics = evaluateModel(y_test_original, y_bagging_pred_original, "BaggingClassifier")
    test_results.append(bagging_metrics)

    voting_clf = create_voting_classifier(X_train_resampled, y_train_resampled)
    y_voting_pred = voting_clf.predict(X_test)
    y_voting_pred_original = label_encoder.inverse_transform(y_voting_pred)

    voting_metrics = evaluateModel(y_test_original, y_voting_pred_original, "VotingClassifier")
    test_results.append(voting_metrics)

    print("\n===== Implementing Stacked Model (Example 6/7 approach) =====")

    model_list = getUnfitModels()
    base_models = []

    for model in model_list:
        model.fit(X_train_resampled, y_train_resampled)
        base_models.append(model)

    dfPredictions = pd.DataFrame()
    for i, model in enumerate(base_models):
        predictions = model.predict_proba(X_train_resampled)
        for j in range(predictions.shape[1]):
            colName = f"model_{i}_class_{j}"
            dfPredictions[colName] = predictions[:, j]

    stacked_model = fitStackedModel(dfPredictions, y_train_resampled)

    dfTestPredictions = pd.DataFrame()
    for i, model in enumerate(base_models):
        test_predictions = model.predict_proba(X_test)
        for j in range(test_predictions.shape[1]):
            colName = f"model_{i}_class_{j}"
            dfTestPredictions[colName] = test_predictions[:, j]

    y_stacked_pred = stacked_model.predict(dfTestPredictions)
    y_stacked_pred_original = label_encoder.inverse_transform(y_stacked_pred)

    stacked_metrics = evaluateModel(y_test_original, y_stacked_pred_original, "StackedModel")
    test_results.append(stacked_metrics)

    test_df = pd.DataFrame(test_results)
    test_df = test_df.sort_values('f1_macro', ascending=False)

    print("\nTest set evaluation results summary:")
    print(test_df.to_string(index=False))

    best_model = test_df.iloc[0]['model_name']
    print(f"\nBest model on test data: {best_model}")
    print(f"  F1 Score: {test_df.iloc[0]['f1_macro']:.4f}")
    print(f"  Accuracy: {test_df.iloc[0]['accuracy']:.4f}")

    if best_model == "StackedModel":
        print("\nDetailed Classification Report for Stacked Model:")
        print(classification_report(y_test_original, y_stacked_pred_original))
    elif best_model == "BaggingClassifier":
        print("\nDetailed Classification Report for Bagging Classifier:")
        print(classification_report(y_test_original, y_bagging_pred_original))
    elif best_model == "VotingClassifier":
        print("\nDetailed Classification Report for Voting Classifier:")
        print(classification_report(y_test_original, y_voting_pred_original))
    else:
        print("\nDetailed Classification Report for Logistic Regression:")
        print(classification_report(y_test_original, y_lr_pred_original))

    return test_df

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('cleaned_cat_outcomes.csv')
        print(f"Dataset loaded with shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Could not find 'cleaned_cat_outcomes.csv'")
        print("Please run data_loading_and_cleaning.py first or ensure the file path is correct.")
        return

    print("\nAdding engineered features...")
    df = create_additional_features(df)

    target_cols = ['outcome_type_grouped', 'outcome_type', 'outcome_type_original', 'outcome_type_encoded']
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    categorical_cols = [col for col in categorical_cols if col not in target_cols]
    numerical_cols = [col for col in numerical_cols if col not in target_cols]

    print("\nCategorical features:")
    print(categorical_cols)
    print("\nNumerical features:")
    print(numerical_cols)

    X = df.drop(target_cols, axis=1)
    y = df['outcome_type_grouped']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nTarget class encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  Class {i}: {label}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=RANDOM_SEED, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\nSetting up preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    print("Applying preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed feature space shape: {X_train_processed.shape}")

    print("\nPerforming cross-validation with required format...")

    if scipy.sparse.issparse(X_train_processed):
        X_train_processed_dense = X_train_processed.toarray()
    else:
        X_train_processed_dense = X_train_processed

    if scipy.sparse.issparse(X_val_processed):
        X_val_processed_dense = X_val_processed.toarray()
    else:
        X_val_processed_dense = X_val_processed

    X_train_val = np.vstack((X_train_processed_dense, X_val_processed_dense))
    y_train_val = np.concatenate((y_train, y_val))

    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"X_val_processed shape: {X_val_processed.shape}")
    print(f"X_train_val shape: {X_train_val.shape}")
    print(f"y_train_val shape: {y_train_val.shape}")

    cv_results = perform_kfold_cross_validation(X_train_val, y_train_val, label_encoder, n_splits=5)

    print("\nTraining final models and evaluating on test set...")
    if scipy.sparse.issparse(X_train_processed):
        X_train_processed = X_train_processed.toarray()
    if scipy.sparse.issparse(X_test_processed):
        X_test_processed = X_test_processed.toarray()

    test_results = train_and_evaluate_models_on_test(
        X_train_processed, y_train,
        X_test_processed, y_test,
        label_encoder
    )

    print("\nAnalysis complete!")
    return test_results

if __name__ == "__main__":
    main()