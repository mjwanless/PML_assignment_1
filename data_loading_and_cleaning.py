import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

file_path = 'aac_shelter_cat_outcome_eng.csv'
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values count per column:")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / df.shape[0]) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1)
missing_data.columns = ['Count', 'Percent']
print(missing_data[missing_data['Count'] > 0].sort_values('Count', ascending=False))

print("\nDistribution of outcome types:")
outcome_counts = df['outcome_type'].value_counts()
outcome_percent = df['outcome_type'].value_counts(normalize=True) * 100
outcome_distribution = pd.concat([outcome_counts, outcome_percent], axis=1)
outcome_distribution.columns = ['Count', 'Percent']
print(outcome_distribution)

columns_to_drop = [
    'animal_id',
    'animal_type',
    'breed2',
    'monthyear',
    'count',
    'Period Range',
    'Periods',
    'outcome_age_(years)',
    'dob_monthyear',
    'age_group',
    'sex_age_outcome',
    'breed',
    'coat',
    'name',
    'color',
    'date_of_birth',
    'datetime',
    'age_upon_outcome'
]

df_cleaned = df.drop(columns=columns_to_drop)

print(f"\nDataset shape after dropping columns: {df_cleaned.shape}")
print("\nRemaining columns:")
print(df_cleaned.columns.tolist())

print("\nFirst 5 rows of cleaned dataset:")
print(df_cleaned.head())

print("\nRemaining columns with missing values:")
missing_data = df_cleaned.isnull().sum()
print(missing_data[missing_data > 0])

df_cleaned = df_cleaned.copy()

df_cleaned['outcome_subtype'] = df_cleaned['outcome_subtype'].fillna('None')

df_cleaned = df_cleaned.dropna(subset=['outcome_type'])

df_cleaned['coat_pattern'] = df_cleaned['coat_pattern'].fillna('Unknown')

df_cleaned['color2'] = df_cleaned['color2'].fillna('None')

print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum().sum())

print(f"\nFinal dataset shape: {df_cleaned.shape}")

print("\nFirst 20 rows of final cleaned dataset:")
print(df_cleaned.head(20))

def categorize_age(days):
    if days < 30:
        return "newborn_kitten"
    elif days < 180:
        return "kitten"
    elif days < 365:
        return "juvenile"
    elif days < 3650:
        return "adult"
    else:
        return "senior"

df_cleaned['age_category'] = df_cleaned['outcome_age_(days)'].apply(categorize_age)

def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

df_cleaned['season'] = df_cleaned['outcome_month'].apply(get_season)

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df_cleaned['time_of_day'] = df_cleaned['outcome_hour'].apply(get_time_of_day)

df_cleaned['cfa_breed'] = df_cleaned['cfa_breed'].astype(bool)
df_cleaned['domestic_breed'] = df_cleaned['domestic_breed'].astype(bool)

categorical_columns = [
    'outcome_subtype', 'outcome_type', 'sex_upon_outcome', 'sex',
    'Spay/Neuter', 'Cat/Kitten (outcome)', 'outcome_weekday',
    'breed1', 'coat_pattern', 'color1', 'color2',
    'age_category', 'season', 'time_of_day'
]

for col in categorical_columns:
    df_cleaned[col] = df_cleaned[col].astype('category')

from sklearn.preprocessing import LabelEncoder

df_cleaned['outcome_type_original'] = df_cleaned['outcome_type']

outcome_encoder = LabelEncoder()
df_cleaned['outcome_type_encoded'] = outcome_encoder.fit_transform(df_cleaned['outcome_type'])

print("\nOutcome Type Encoding:")
for i, category in enumerate(outcome_encoder.classes_):
    print(f"{category} -> {i}")

print("\nFeature Engineering Complete")
print(f"Final dataset shape: {df_cleaned.shape}")
print("\nNew column data types:")
print(df_cleaned.dtypes)

print("\nAge Category Distribution:")
print(df_cleaned['age_category'].value_counts())

print("\nSeason Distribution:")
print(df_cleaned['season'].value_counts())

print("\nTime of Day Distribution:")
print(df_cleaned['time_of_day'].value_counts())

print("\nSample of the final processed dataset:")
print(df_cleaned.head(30))

print("\n\n===== INVESTIGATING AGE OUTLIERS =====")
print("Cats with extremely high ages (> 15 years):")
old_cats = df_cleaned[df_cleaned['outcome_age_(days)'] > 5475]
print(f"Number of very old cats: {len(old_cats)}")
print(old_cats[['outcome_age_(days)', 'breed1', 'outcome_type', 'sex_upon_outcome']].head(10))

print("\nAge statistics (in years):")
age_years = df_cleaned['outcome_age_(days)'] / 365
print(f"Mean: {age_years.mean():.2f}")
print(f"Median: {age_years.median():.2f}")
print(f"95th percentile: {age_years.quantile(0.95):.2f}")
print(f"99th percentile: {age_years.quantile(0.99):.2f}")

print("\n\n===== EXAMINING CLASS IMBALANCE =====")
outcome_counts = df_cleaned['outcome_type'].value_counts()
outcome_percents = df_cleaned['outcome_type'].value_counts(normalize=True) * 100

print("\nDetailed class distribution:")
for outcome, count in outcome_counts.items():
    print(f"{outcome}: {count} ({outcome_percents[outcome]:.2f}%)")

print("\nPotential class grouping strategy:")
print("Main classes: 'Transfer', 'Adoption', 'Euthanasia', 'Return to Owner'")
print("Combined class 'Other': 'Died', 'Rto-Adopt', 'Missing', 'Disposal'")
combined_counts = outcome_counts.copy()
other_classes = ['Died', 'Rto-Adopt', 'Missing', 'Disposal']
other_count = sum(combined_counts[cls] for cls in other_classes if cls in combined_counts)
for cls in other_classes:
    if cls in combined_counts:
        combined_counts = combined_counts.drop(cls)
combined_counts['Other'] = other_count

combined_percents = combined_counts / combined_counts.sum() * 100
print("\nDistribution after combining rare classes:")
for outcome, count in combined_counts.items():
    print(f"{outcome}: {count} ({combined_percents[outcome]:.2f}%)")

df_cleaned['outcome_type_grouped'] = df_cleaned['outcome_type'].apply(
    lambda x: 'Other' if x in other_classes else x
)
df_cleaned['outcome_type_grouped'] = df_cleaned['outcome_type_grouped'].astype('category')

print("\nVerifying the new grouped outcome distribution:")
grouped_counts = df_cleaned['outcome_type_grouped'].value_counts()
grouped_percents = df_cleaned['outcome_type_grouped'].value_counts(normalize=True) * 100
for outcome, count in grouped_counts.items():
    print(f"{outcome}: {count} ({grouped_percents[outcome]:.2f}%)")

output_file = 'cleaned_cat_outcomes.csv'
df_cleaned.to_csv(output_file, index=False)