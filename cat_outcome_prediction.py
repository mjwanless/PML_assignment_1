# Cat Shelter Outcome Prediction - COMP4948 Assignment
# Implements all required models: LogisticRegression, BaggingClassifier,
# VotingClassifier, and Stacked models with proper cross-validation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_additional_features(df):
    """Adds engineered features to improve predictive power"""
    # Weekend/weekday feature
    if 'outcome_weekday' in df.columns:
        df['is_weekend'] = df['outcome_weekday'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    # Kitten season indicator
    if 'outcome_month' in df.columns:
        df['is_kitten_season'] = df['outcome_month'].apply(lambda x: 1 if x in [3, 4, 5, 6] else 0)

    # Business hours indicator
    if 'outcome_hour' in df.columns:
        df['during_business_hours'] = df['outcome_hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)

    # Age categories with more granularity
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
    """Applies SMOTE to handle class imbalance in training data"""
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Display class distribution before and after
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
    """Returns a list of unfitted classifier models"""
    models = list()
    models.append(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED))
    models.append(DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=RANDOM_SEED))
    models.append(AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED))
    models.append(RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=RANDOM_SEED))
    models.append(SVC(probability=True, class_weight='balanced', random_state=RANDOM_SEED))
    return models

def evaluateModel(y_true, predictions, model_name):
    """Evaluates model using consistent metrics"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, predictions)
    f1_macro = f1_score(y_true, predictions, average='macro')
    precision = precision_score(y_true, predictions, average='macro')
    recall = recall_score(y_true, predictions, average='macro')

    # Print metrics
    print(f"\n*** {model_name} Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")

    # Return metrics for comparison
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall
    }

def fitBaseModels(X_train, y_train, X_val, models):
    """Fits base models and returns their predictions for the stacked model"""
    dfPredictions = pd.DataFrame()

    # Fit each base model and store its predictions
    for i, model in enumerate(models):
        model_name = model.__class__.__name__
        print(f"Training base model {i+1}/{len(models)}: {model_name}")

        model.fit(X_train, y_train)

        # For stacked model, we use probabilities to get more information
        predictions = model.predict_proba(X_val)

        # Add each class probability as a separate column
        for j in range(predictions.shape[1]):
            colName = f"model_{i}_class_{j}"
            dfPredictions[colName] = predictions[:, j]

    return dfPredictions, models

def fitStackedModel(X, y):
    """Fits a stacked model (meta-learner) using base model predictions"""
    print("Training stacked model (meta-learner)...")
    meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    meta_learner.fit(X, y)
    return meta_learner

def create_bagged_model(X_train, y_train):
    """Creates and trains a bagged classifier with LogisticRegression"""
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
    """Creates and trains a voting classifier with multiple model types"""
    print("\n===== Training Voting Classifier =====")

    # Create base models
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=RANDOM_SEED)
    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=RANDOM_SEED)

    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('dt', dt)
        ],
        voting='soft'
    )

    print("Training VotingClassifier...")
    voting_clf.fit(X_train, y_train)
    return voting_clf

def perform_kfold_cross_validation(X, y, label_encoder, n_splits=5):
    """Performs k-fold cross-validation with the required format in the assignment"""
    print("\n===== Cross-Validation with Required KFold Format =====")

    # Create models
    models = getUnfitModels()
    model_names = [model.__class__.__name__ for model in models]
    model_names.extend(['BaggingClassifier', 'VotingClassifier', 'StackedModel'])

    # Create k-fold object with the exact format from the assignment
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    # Storage for fold results
    fold_results = {name: {'accuracy': [], 'f1_macro': [], 'precision': [], 'recall': []}
                    for name in model_names}

    # Loop through folds using the required format from the assignment
    for fold, (train_index, test_index) in enumerate(kfold.split(X), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        # Split data for this fold
        X_fold_train, X_fold_test = X[train_index], X[test_index]
        y_fold_train, y_fold_test = y[train_index], y[test_index]

        # Apply SMOTE to the training data in this fold
        X_fold_train_resampled, y_fold_train_resampled = apply_smote_balancing(X_fold_train, y_fold_train)

        # Train and evaluate regular models
        for i, model in enumerate(models):
            model_name = model.__class__.__name__
            print(f"Training {model_name}...")

            # Train model
            model.fit(X_fold_train_resampled, y_fold_train_resampled)

            # Evaluate on test data
            y_pred = model.predict(X_fold_test)

            # Convert to original labels for better interpretability
            y_pred_original = label_encoder.inverse_transform(y_pred)
            y_test_original = label_encoder.inverse_transform(y_fold_test)

            # Get metrics
            metrics = evaluateModel(y_test_original, y_pred_original, model_name)

            # Store results
            fold_results[model_name]['accuracy'].append(metrics['accuracy'])
            fold_results[model_name]['f1_macro'].append(metrics['f1_macro'])
            fold_results[model_name]['precision'].append(metrics['precision'])
            fold_results[model_name]['recall'].append(metrics['recall'])

        # Train and evaluate bagging classifier
        bagging_clf = create_bagged_model(X_fold_train_resampled, y_fold_train_resampled)
        y_bagging_pred = bagging_clf.predict(X_fold_test)
        y_bagging_pred_original = label_encoder.inverse_transform(y_bagging_pred)

        bagging_metrics = evaluateModel(y_test_original, y_bagging_pred_original, "BaggingClassifier")
        fold_results['BaggingClassifier']['accuracy'].append(bagging_metrics['accuracy'])
        fold_results['BaggingClassifier']['f1_macro'].append(bagging_metrics['f1_macro'])
        fold_results['BaggingClassifier']['precision'].append(bagging_metrics['precision'])
        fold_results['BaggingClassifier']['recall'].append(bagging_metrics['recall'])

        # Train and evaluate voting classifier
        voting_clf = create_voting_classifier(X_fold_train_resampled, y_fold_train_resampled)
        y_voting_pred = voting_clf.predict(X_fold_test)
        y_voting_pred_original = label_encoder.inverse_transform(y_voting_pred)

        voting_metrics = evaluateModel(y_test_original, y_voting_pred_original, "VotingClassifier")
        fold_results['VotingClassifier']['accuracy'].append(voting_metrics['accuracy'])
        fold_results['VotingClassifier']['f1_macro'].append(voting_metrics['f1_macro'])
        fold_results['VotingClassifier']['precision'].append(voting_metrics['precision'])
        fold_results['VotingClassifier']['recall'].append(voting_metrics['recall'])

        # Create and evaluate stacked model
        # First get predictions from base models for meta-learner
        dfPredictions, fitted_models = fitBaseModels(X_fold_train_resampled, y_fold_train_resampled, X_fold_train_resampled, models.copy())
        stacked_model = fitStackedModel(dfPredictions, y_fold_train_resampled)

        # Get predictions from base models for test data
        dfTestPredictions = pd.DataFrame()
        for i, model in enumerate(fitted_models):
            test_predictions = model.predict_proba(X_fold_test)
            for j in range(test_predictions.shape[1]):
                colName = f"model_{i}_class_{j}"
                dfTestPredictions[colName] = test_predictions[:, j]

        # Evaluate stacked model
        y_stacked_pred = stacked_model.predict(dfTestPredictions)
        y_stacked_pred_original = label_encoder.inverse_transform(y_stacked_pred)

        stacked_metrics = evaluateModel(y_test_original, y_stacked_pred_original, "StackedModel")
        fold_results['StackedModel']['accuracy'].append(stacked_metrics['accuracy'])
        fold_results['StackedModel']['f1_macro'].append(stacked_metrics['f1_macro'])
        fold_results['StackedModel']['precision'].append(stacked_metrics['precision'])
        fold_results['StackedModel']['recall'].append(stacked_metrics['recall'])

    # Calculate and display average performance across folds
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

    # Convert to DataFrame and sort by F1 score
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('f1_macro_mean', ascending=False)

    print("\nModel performance summary (averaged across all folds):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_df.to_string(index=False))

    # Identify best model
    best_model = summary_df.iloc[0]['model_name']
    print(f"\nBest model based on F1 score: {best_model}")
    print(f"  Average F1 Score: {summary_df.iloc[0]['f1_macro_mean']:.4f} (±{summary_df.iloc[0]['f1_macro_std']:.4f})")

    return summary_df

def train_and_evaluate_models_on_test(X_train, y_train, X_test, y_test, label_encoder):
    """Trains models on training data and evaluates on test data"""
    print("\n===== Final Model Training and Evaluation on Test Set =====")

    # Apply SMOTE for class balancing
    X_train_resampled, y_train_resampled = apply_smote_balancing(X_train, y_train)

    # Get base models
    models = getUnfitModels()

    # Train and evaluate base models
    test_results = []
    fitted_models = []

    for model in models:
        model_name = model.__class__.__name__
        print(f"\nTraining {model_name}...")

        # Train model
        model.fit(X_train_resampled, y_train_resampled)
        fitted_models.append(model)

        # Evaluate on test data
        y_pred = model.predict(X_test)

        # Convert predictions to original labels
        y_pred_original = label_encoder.inverse_transform(y_pred)
        y_test_original = label_encoder.inverse_transform(y_test)

        # Get metrics
        metrics = evaluateModel(y_test_original, y_pred_original, model_name)
        test_results.append(metrics)

    # Train and evaluate bagging classifier
    bagging_clf = create_bagged_model(X_train_resampled, y_train_resampled)
    y_bagging_pred = bagging_clf.predict(X_test)
    y_bagging_pred_original = label_encoder.inverse_transform(y_bagging_pred)
    y_test_original = label_encoder.inverse_transform(y_test)

    bagging_metrics = evaluateModel(y_test_original, y_bagging_pred_original, "BaggingClassifier")
    test_results.append(bagging_metrics)

    # Train and evaluate voting classifier
    voting_clf = create_voting_classifier(X_train_resampled, y_train_resampled)
    y_voting_pred = voting_clf.predict(X_test)
    y_voting_pred_original = label_encoder.inverse_transform(y_voting_pred)

    voting_metrics = evaluateModel(y_test_original, y_voting_pred_original, "VotingClassifier")
    test_results.append(voting_metrics)

    # Create and evaluate stacked model - Using Example 6/7 approach from week 3
    print("\n===== Implementing Stacked Model (Example 6/7 approach) =====")

    # Get predictions from base models for stacked model training
    dfPredictions, _ = fitBaseModels(X_train_resampled, y_train_resampled, X_train_resampled, fitted_models)
    stacked_model = fitStackedModel(dfPredictions, y_train_resampled)

    # Get test predictions from base models for stacked model
    dfTestPredictions = pd.DataFrame()
    for i, model in enumerate(fitted_models):
        test_predictions = model.predict_proba(X_test)
        for j in range(test_predictions.shape[1]):
            colName = f"model_{i}_class_{j}"
            dfTestPredictions[colName] = test_predictions[:, j]

    # Evaluate stacked model
    y_stacked_pred = stacked_model.predict(dfTestPredictions)
    y_stacked_pred_original = label_encoder.inverse_transform(y_stacked_pred)

    stacked_metrics = evaluateModel(y_test_original, y_stacked_pred_original, "StackedModel")
    test_results.append(stacked_metrics)

    # Convert to DataFrame and sort by F1 score
    test_df = pd.DataFrame(test_results)
    test_df = test_df.sort_values('f1_macro', ascending=False)

    print("\nTest set evaluation results summary:")
    print(test_df.to_string(index=False))

    # Identify best model
    best_model = test_df.iloc[0]['model_name']
    print(f"\nBest model on test data: {best_model}")
    print(f"  F1 Score: {test_df.iloc[0]['f1_macro']:.4f}")
    print(f"  Accuracy: {test_df.iloc[0]['accuracy']:.4f}")

    # Print classification report for the best model
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
        # Find the corresponding model in fitted_models
        for i, model in enumerate(fitted_models):
            if model.__class__.__name__ == best_model:
                best_idx = i
                break

        y_best_pred = fitted_models[best_idx].predict(X_test)
        y_best_pred_original = label_encoder.inverse_transform(y_best_pred)

        print(f"\nDetailed Classification Report for {best_model}:")
        print(classification_report(y_test_original, y_best_pred_original))

    return test_df

def feature_importance_analysis(X_train, y_train, feature_names):
    """Analyzes feature importance using RandomForest"""
    print("\n===== Feature Importance Analysis =====")

    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_

    # Create a dataframe of feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # Print top 20 features
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    return feature_importance

def main():
    """Main function to execute the entire pipeline"""
    print("Loading data...")
    try:
        df = pd.read_csv('cleaned_cat_outcomes.csv')
        print(f"Dataset loaded with shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Could not find 'cleaned_cat_outcomes.csv'")
        print("Please run data_loading_and_cleaning.py first or ensure the file path is correct.")
        return

    # Add engineered features
    print("\nAdding engineered features...")
    df = create_additional_features(df)

    # Identify feature columns and target column
    target_cols = ['outcome_type_grouped', 'outcome_type', 'outcome_type_original', 'outcome_type_encoded']
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove target columns from feature lists
    categorical_cols = [col for col in categorical_cols if col not in target_cols]
    numerical_cols = [col for col in numerical_cols if col not in target_cols]

    print("\nCategorical features:")
    print(categorical_cols)
    print("\nNumerical features:")
    print(numerical_cols)

    # Split features and target
    X = df.drop(target_cols, axis=1)
    y = df['outcome_type_grouped']

    # Create a label encoder for the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Show class encoding
    print("\nTarget class encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  Class {i}: {label}")

    # Split data into train, validation, and test sets (60%, 20%, 20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=RANDOM_SEED, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create preprocessing pipeline
    print("\nSetting up preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Apply preprocessing
    print("Applying preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed feature space shape: {X_train_processed.shape}")

    # Get feature names after one-hot encoding for better interpretability
    # This is complex due to ColumnTransformer, so we'll use indices
    feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    # Analyze feature importance
    feature_importance = feature_importance_analysis(X_train_processed, y_train, feature_names)

    # Perform cross-validation with required format
    print("\nPerforming cross-validation with required format...")
    # Combine train and validation sets for cross-validation
    X_train_val = np.vstack((X_train_processed, X_val_processed))
    y_train_val = np.concatenate((y_train, y_val))

    cv_results = perform_kfold_cross_validation(X_train_val, y_train_val, label_encoder, n_splits=5)

    # Train final models and evaluate on test set
    print("\nTraining final models and evaluating on test set...")
    test_results = train_and_evaluate_models_on_test(
        X_train_processed, y_train,
        X_test_processed, y_test,
        label_encoder
    )

    print("\nAnalysis complete!")
    return test_results

if __name__ == "__main__":
    main()