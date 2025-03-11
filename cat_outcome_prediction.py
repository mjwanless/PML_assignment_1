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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
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

def perform_eda(df, target_column, numerical_cols, categorical_cols):
    print("\n===== Exploratory Data Analysis =====")

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")

    plt.figure(figsize=(12, 6))
    target_counts = df[target_column].value_counts()
    ax = sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title('Distribution of Cat Outcomes', fontsize=16)
    plt.xlabel('Outcome Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    total = len(df)
    for i, count in enumerate(target_counts.values):
        percentage = 100 * count / total
        ax.text(i, count + 50, f'{percentage:.1f}%', ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    X_sample = df.drop(['outcome_type_grouped', 'outcome_type', 'outcome_type_original', 'outcome_type_encoded'], axis=1, errors='ignore')
    y_sample = df[target_column]

    cat_encoder = OneHotEncoder(handle_unknown='ignore')
    X_cat = cat_encoder.fit_transform(X_sample[categorical_cols])

    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        cat_values = cat_encoder.categories_[i]
        cat_feature_names.extend([f"{col}_{val}" for val in cat_values])

    X_num = X_sample[numerical_cols].values
    X_combined = np.hstack([X_num, X_cat.toarray()])
    feature_names = numerical_cols + cat_feature_names

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_sample)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    model.fit(X_combined, y_encoded)

    importances = np.abs(model.coef_).mean(axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(14, 8))
    top_n = 15
    top_indices = indices[:top_n]
    sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices])
    plt.title(f'Top {top_n} Most Important Features', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()

    top_features = [feature_names[i] for i in indices[:10]]
    print(f"\nTop 10 most important features: {top_features}")

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols[:8]):
        plt.subplot(2, 4, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

    num_top_features = [f for f in top_features if f in numerical_cols][:4]

    if num_top_features:
        fig = plt.figure(figsize=(16, 12))
        for i, feature in enumerate(num_top_features):
            if feature in df.columns:
                plt.subplot(2, 2, i+1)
                sns.boxplot(x=target_column, y=feature, data=df)
                plt.title(f'{feature} by Outcome Type', fontsize=14)
                plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    cat_top_features = [f for f in categorical_cols if any(f in feat for feat in top_features)][:4]

    if cat_top_features:
        for feature in cat_top_features:
            plt.figure(figsize=(14, 8))
            cross_tab = pd.crosstab(df[feature], df[target_column], normalize='index') * 100
            cross_tab.plot(kind='bar', stacked=True)
            plt.title(f'Outcome Distribution by {feature}', fontsize=16)
            plt.xlabel(feature, fontsize=14)
            plt.ylabel('Percentage', fontsize=14)
            plt.legend(title='Outcome Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    if 'outcome_age_(days)' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.histplot(df['outcome_age_(days)'], bins=50, kde=True)
        plt.title('Distribution of Cat Ages', fontsize=16)
        plt.xlabel('Age (days)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 8))
        sns.boxplot(x=target_column, y='outcome_age_(days)', data=df)
        plt.title('Cat Age by Outcome Type', fontsize=16)
        plt.xlabel('Outcome Type', fontsize=14)
        plt.ylabel('Age (days)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if 'age_months_category' in df.columns:
            plt.figure(figsize=(14, 8))
            order = ['0-2months', '2-6months', '6-12months', '1-3years', '3-7years', '7+years']
            age_outcome = pd.crosstab(df['age_months_category'], df[target_column], normalize='index') * 100
            age_outcome = age_outcome.reindex(order)
            age_outcome.plot(kind='bar', stacked=True)
            plt.title('Outcome Distribution by Age Category', fontsize=16)
            plt.xlabel('Age Category', fontsize=14)
            plt.ylabel('Percentage', fontsize=14)
            plt.legend(title='Outcome Type')
            plt.tight_layout()
            plt.show()

    if 'outcome_month' in df.columns and 'outcome_weekday' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        month_order = range(1, 13)
        outcomes_by_month = pd.crosstab(df['outcome_month'], df[target_column])
        outcomes_by_month = outcomes_by_month.reindex(month_order)

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        outcomes_by_month.index = [month_names[i-1] for i in outcomes_by_month.index]

        outcomes_by_month.plot(kind='bar', stacked=True, ax=ax1)
        ax1.set_title('Outcomes by Month', fontsize=16)
        ax1.set_xlabel('Month', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)

        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        outcomes_by_weekday = pd.crosstab(df['outcome_weekday'], df[target_column])
        outcomes_by_weekday = outcomes_by_weekday.reindex(weekday_order)
        outcomes_by_weekday.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title('Outcomes by Weekday', fontsize=16)
        ax2.set_xlabel('Weekday', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)

        plt.tight_layout()
        plt.show()

    if len(numerical_cols) > 1:
        plt.figure(figsize=(14, 12))
        corr_matrix = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features', fontsize=16)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(20, 25))
    gs = GridSpec(5, 2)

    ax1 = plt.subplot(gs[0, :])
    target_counts = df[target_column].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax1)
    ax1.set_title('Distribution of Cat Outcomes', fontsize=16)
    ax1.set_xlabel('Outcome Type', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = plt.subplot(gs[1, :])
    top_n = 10
    top_indices = indices[:top_n]
    sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices], ax=ax2)
    ax2.set_title(f'Top {top_n} Most Important Features', fontsize=16)
    ax2.set_xlabel('Importance Score', fontsize=14)
    ax2.set_ylabel('Feature', fontsize=14)

    if 'outcome_age_(days)' in df.columns:
        ax3 = plt.subplot(gs[2, 0])
        sns.histplot(df['outcome_age_(days)'], bins=30, kde=True, ax=ax3)
        ax3.set_title('Distribution of Cat Ages', fontsize=16)
        ax3.set_xlabel('Age (days)', fontsize=14)
        ax3.set_ylabel('Count', fontsize=14)

        ax4 = plt.subplot(gs[2, 1])
        sns.boxplot(x=target_column, y='outcome_age_(days)', data=df, ax=ax4)
        ax4.set_title('Cat Age by Outcome Type', fontsize=16)
        ax4.set_xlabel('Outcome Type', fontsize=14)
        ax4.set_ylabel('Age (days)', fontsize=14)
        ax4.tick_params(axis='x', rotation=45)

    if 'outcome_month' in df.columns and 'outcome_weekday' in df.columns:
        ax5 = plt.subplot(gs[3, 0])
        outcomes_by_month = pd.crosstab(df['outcome_month'], df[target_column])
        outcomes_by_month = outcomes_by_month.reindex(month_order)
        outcomes_by_month.index = [month_names[i-1] for i in outcomes_by_month.index]
        outcomes_by_month.plot(kind='bar', stacked=True, ax=ax5)
        ax5.set_title('Outcomes by Month', fontsize=16)
        ax5.set_xlabel('Month', fontsize=14)
        ax5.set_ylabel('Count', fontsize=14)
        ax5.tick_params(axis='x', rotation=45)

        ax6 = plt.subplot(gs[3, 1])
        outcomes_by_weekday = pd.crosstab(df['outcome_weekday'], df[target_column])
        outcomes_by_weekday = outcomes_by_weekday.reindex(weekday_order)
        outcomes_by_weekday.plot(kind='bar', stacked=True, ax=ax6)
        ax6.set_title('Outcomes by Weekday', fontsize=16)
        ax6.set_xlabel('Weekday', fontsize=14)
        ax6.set_ylabel('Count', fontsize=14)
        ax6.tick_params(axis='x', rotation=45)

    if cat_top_features and len(cat_top_features) > 0:
        feature = cat_top_features[0]
        ax7 = plt.subplot(gs[4, :])
        cross_tab = pd.crosstab(df[feature], df[target_column], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, ax=ax7)
        ax7.set_title(f'Outcome Distribution by {feature}', fontsize=16)
        ax7.set_xlabel(feature, fontsize=14)
        ax7.set_ylabel('Percentage', fontsize=14)
        ax7.legend(title='Outcome Type')
        ax7.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print("\n***** Key EDA Findings *****")
    print(f"• Total number of cats in the dataset: {len(df)}")
    print(f"• Most common outcome: {df[target_column].value_counts().index[0]} ({df[target_column].value_counts().iloc[0]} cats)")
    print(f"• Most important features affecting cat outcomes: {', '.join(top_features[:5])}")

    if 'age_months_category' in df.columns:
        print(f"• Age distribution: {df['age_months_category'].value_counts().to_dict()}")

    if 'is_weekend' in df.columns:
        weekend_pct = df['is_weekend'].mean() * 100
        print(f"• Percentage of outcomes occurring on weekends: {weekend_pct:.1f}%")

    if 'is_kitten_season' in df.columns:
        kitten_season_pct = df['is_kitten_season'].mean() * 100
        print(f"• Percentage of outcomes during kitten season (Mar-Jun): {kitten_season_pct:.1f}%")

    return top_features

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

def create_comparison_report(cv_results, test_results):
    print("\n===== Model Comparison Report =====")

    model_names = cv_results['model_name'].tolist()

    comparison_data = []

    for model in model_names:
        cv_model_row = cv_results[cv_results['model_name'] == model].iloc[0]
        test_model_row = test_results[test_results['model_name'] == model].iloc[0]

        model_data = {
            'Model': model,
            'CV F1 Score': f"{cv_model_row['f1_macro_mean']:.4f} ± {cv_model_row['f1_macro_std']:.4f}",
            'Test F1 Score': f"{test_model_row['f1_macro']:.4f}",
            'CV Accuracy': f"{cv_model_row['accuracy_mean']:.4f} ± {cv_model_row['accuracy_std']:.4f}",
            'Test Accuracy': f"{test_model_row['accuracy']:.4f}",
            'CV Precision': f"{cv_model_row['precision_mean']:.4f} ± {cv_model_row['precision_std']:.4f}",
            'Test Precision': f"{test_model_row['precision']:.4f}",
            'CV Recall': f"{cv_model_row['recall_mean']:.4f} ± {cv_model_row['recall_std']:.4f}",
            'Test Recall': f"{test_model_row['recall']:.4f}",
        }
        comparison_data.append(model_data)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test F1 Score', ascending=False)

    print("\nComprehensive Model Comparison Table:")
    print(comparison_df.to_string(index=False))

    plt.figure(figsize=(14, 10))

    models = comparison_df['Model'].tolist()
    cv_f1 = [float(x.split(' ±')[0]) for x in comparison_df['CV F1 Score']]
    test_f1 = [float(x) for x in comparison_df['Test F1 Score']]

    x = np.arange(len(models))
    width = 0.35

    plt.subplot(2, 2, 1)
    plt.bar(x - width/2, cv_f1, width, label='Cross-Validation')
    plt.bar(x + width/2, test_f1, width, label='Test')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()

    cv_acc = [float(x.split(' ±')[0]) for x in comparison_df['CV Accuracy']]
    test_acc = [float(x) for x in comparison_df['Test Accuracy']]

    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, cv_acc, width, label='Cross-Validation')
    plt.bar(x + width/2, test_acc, width, label='Test')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()

    cv_prec = [float(x.split(' ±')[0]) for x in comparison_df['CV Precision']]
    test_prec = [float(x) for x in comparison_df['Test Precision']]

    plt.subplot(2, 2, 3)
    plt.bar(x - width/2, cv_prec, width, label='Cross-Validation')
    plt.bar(x + width/2, test_prec, width, label='Test')
    plt.xlabel('Model')
    plt.ylabel('Precision')
    plt.title('Precision Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()

    cv_recall = [float(x.split(' ±')[0]) for x in comparison_df['CV Recall']]
    test_recall = [float(x) for x in comparison_df['Test Recall']]

    plt.subplot(2, 2, 4)
    plt.bar(x - width/2, cv_recall, width, label='Cross-Validation')
    plt.bar(x + width/2, test_recall, width, label='Test')
    plt.xlabel('Model')
    plt.ylabel('Recall')
    plt.title('Recall Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.show()

    best_model = comparison_df.iloc[0]['Model']

    metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    test_values = [
        float(comparison_df.iloc[0]['Test F1 Score']),
        float(comparison_df.iloc[0]['Test Accuracy']),
        float(comparison_df.iloc[0]['Test Precision']),
        float(comparison_df.iloc[0]['Test Recall'])
    ]

    plt.figure(figsize=(10, 10))

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    values = test_values + test_values[:1]

    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2, label=best_model)
    ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)

    ax.set_ylim(0, 1)

    plt.title(f'Performance Metrics for {best_model}', size=15, y=1.1)
    plt.tight_layout()
    plt.show()

    print("\n***** Model Comparison Summary *****")
    print(f"Best performing model: {best_model}")
    print(f"F1 Score on test data: {comparison_df.iloc[0]['Test F1 Score']}")
    print(f"Performance across cross-validation: {comparison_df.iloc[0]['CV F1 Score']}")
    print("\nModel comparison summary table (all metrics):")
    print(comparison_df.to_string(index=False))

    return comparison_df

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

    print("\nPerforming exploratory data analysis...")
    top_features = perform_eda(df, 'outcome_type_grouped', numerical_cols, categorical_cols)

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

    print("\nCreating comprehensive model comparison report...")
    comparison_results = create_comparison_report(cv_results, test_results)

    print("\nAnalysis complete!")
    return test_results

if __name__ == "__main__":
    main()