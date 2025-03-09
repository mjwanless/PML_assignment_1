import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options to show more columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

# Load the data
file_path = 'aac_shelter_cat_outcome_eng.csv'  # Update this path if needed
df = pd.read_csv(file_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset info:")
print(df.info())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check missing values
print("\nMissing values count per column:")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / df.shape[0]) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1)
missing_data.columns = ['Count', 'Percent']
print(missing_data[missing_data['Count'] > 0].sort_values('Count', ascending=False))

# Check distribution of outcome types
print("\nDistribution of outcome types:")
outcome_counts = df['outcome_type'].value_counts()
outcome_percent = df['outcome_type'].value_counts(normalize=True) * 100
outcome_distribution = pd.concat([outcome_counts, outcome_percent], axis=1)
outcome_distribution.columns = ['Count', 'Percent']
print(outcome_distribution)

# # Basic visualization of outcome distribution
# plt.figure(figsize=(10, 6))
# sns.countplot(y='outcome_type', data=df, order=df['outcome_type'].value_counts().index)
# plt.title('Distribution of Cat Outcome Types')
# plt.tight_layout()
# plt.show()

# # Check distribution of cat/kitten
# print("\nDistribution by Cat/Kitten status:")
# cat_kitten_counts = df['Cat/Kitten (outcome)'].value_counts()
# cat_kitten_percent = df['Cat/Kitten (outcome)'].value_counts(normalize=True) * 100
# cat_kitten_distribution = pd.concat([cat_kitten_counts, cat_kitten_percent], axis=1)
# cat_kitten_distribution.columns = ['Count', 'Percent']
# print(cat_kitten_distribution)
#
# # Check the relationship between Cat/Kitten and outcome
# print("\nOutcome distribution by Cat/Kitten status:")
# outcome_by_cat_kitten = pd.crosstab(df['Cat/Kitten (outcome)'], df['outcome_type'], normalize='index') * 100
# print(outcome_by_cat_kitten.round(2))
#
# # Visualize the relationship between Cat/Kitten and outcome
# plt.figure(figsize=(12, 6))
# outcome_by_cat_kitten_counts = pd.crosstab(df['Cat/Kitten (outcome)'], df['outcome_type'])
# outcome_by_cat_kitten_counts.plot(kind='bar', stacked=True)
# plt.title('Outcome Types by Cat/Kitten Status')
# plt.ylabel('Count')
# plt.xticks(rotation=0)
# plt.legend(title='Outcome Type')
# plt.tight_layout()
# plt.show()


# Updated list of columns to drop
columns_to_drop = [
    'animal_id',        # Unique identifier, not predictive
    'animal_type',      # All records are "Cat"
    'breed2',           # 99.8% missing
    'monthyear',        # Redundant with datetime
    'count',            # All values are 1
    'Period Range',     # Redundant with other age columns
    'Periods',          # Redundant with other age columns
    'outcome_age_(years)', # Keep days instead for more granularity
    'dob_monthyear',    # Redundant with other date columns
    'age_group',        # Redundant with other age information
    'sex_age_outcome',  # Combination of information we already have
    'breed',            # Redundant with breed1
    'coat',             # Appears to duplicate color1
    'name',             # High missing values (43.4%) and high cardinality
    'color',            # Redundant with color1 and color2
    'date_of_birth',    # We have dob_year and dob_month already
    'datetime',         # We have outcome_year, outcome_month etc. already
    'age_upon_outcome'  # Redundant with outcome_age_(days)
]

# Drop the columns
df_cleaned = df.drop(columns=columns_to_drop)

# Display the shape after dropping columns
print(f"\nDataset shape after dropping columns: {df_cleaned.shape}")
print("\nRemaining columns:")
print(df_cleaned.columns.tolist())

# Preview the cleaned data
print("\nFirst 5 rows of cleaned dataset:")
print(df_cleaned.head())

# Check which columns still have missing values
print("\nRemaining columns with missing values:")
missing_data = df_cleaned.isnull().sum()
print(missing_data[missing_data > 0])

# Handle missing values according to your specified strategies:
# Create a copy to ensure we're working with a fresh DataFrame
df_cleaned = df_cleaned.copy()

# 1. outcome_subtype - replace with "None" (avoiding inplace)
df_cleaned['outcome_subtype'] = df_cleaned['outcome_subtype'].fillna('None')

# 2. outcome_type - drop the 3 rows with missing values
df_cleaned = df_cleaned.dropna(subset=['outcome_type'])

# 3. coat_pattern - replace with "Unknown" (avoiding inplace)
df_cleaned['coat_pattern'] = df_cleaned['coat_pattern'].fillna('Unknown')

# 4. color2 - replace with "None" (avoiding inplace)
df_cleaned['color2'] = df_cleaned['color2'].fillna('None')

# Verify no more missing values
print("\nMissing values after cleaning:")
print(df_cleaned.isnull().sum().sum())

# Display the shape of the final cleaned dataset
print(f"\nFinal dataset shape: {df_cleaned.shape}")

# Preview the cleaned dataset
print("\nFirst 20 rows of final cleaned dataset:")
print(df_cleaned.head(20))

# Continue from your previous data cleaning code

# 1. Feature Engineering

# Create age_category from outcome_age_(days)
def categorize_age(days):
    if days < 30:
        return "newborn_kitten"
    elif days < 180:
        return "kitten"
    elif days < 365:
        return "juvenile"
    elif days < 3650:  # 10 years
        return "adult"
    else:
        return "senior"

df_cleaned['age_category'] = df_cleaned['outcome_age_(days)'].apply(categorize_age)

# Create season from outcome_month
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:  # month in [12, 1, 2]
        return "Winter"

df_cleaned['season'] = df_cleaned['outcome_month'].apply(get_season)

# Create time_of_day from outcome_hour
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:  # 21-23, 0-4
        return "Night"

df_cleaned['time_of_day'] = df_cleaned['outcome_hour'].apply(get_time_of_day)

# 2. Data Type Conversion

# Convert boolean columns to proper boolean type
df_cleaned['cfa_breed'] = df_cleaned['cfa_breed'].astype(bool)
df_cleaned['domestic_breed'] = df_cleaned['domestic_breed'].astype(bool)

# Convert categorical columns to categorical data type
categorical_columns = [
    'outcome_subtype', 'outcome_type', 'sex_upon_outcome', 'sex',
    'Spay/Neuter', 'Cat/Kitten (outcome)', 'outcome_weekday',
    'breed1', 'coat_pattern', 'color1', 'color2',
    'age_category', 'season', 'time_of_day'
]

for col in categorical_columns:
    df_cleaned[col] = df_cleaned[col].astype('category')

# 3. Encode Categorical Variables - we'll do this in a way that avoids high dimensionality

# For target variable (outcome_type), we'll use label encoding
from sklearn.preprocessing import LabelEncoder

# Create a copy of the original target for reference
df_cleaned['outcome_type_original'] = df_cleaned['outcome_type']

# Create the encoder and fit it to the data
outcome_encoder = LabelEncoder()
df_cleaned['outcome_type_encoded'] = outcome_encoder.fit_transform(df_cleaned['outcome_type'])

# Print the mapping for reference
print("\nOutcome Type Encoding:")
for i, category in enumerate(outcome_encoder.classes_):
    print(f"{category} -> {i}")

# For other categorical variables, we'll prepare them for one-hot encoding later during modeling
# by keeping them as categorical data types for now

# Print information about the processed dataset
print("\nFeature Engineering Complete")
print(f"Final dataset shape: {df_cleaned.shape}")
print("\nNew column data types:")
print(df_cleaned.dtypes)

# Display category counts
print("\nAge Category Distribution:")
print(df_cleaned['age_category'].value_counts())

print("\nSeason Distribution:")
print(df_cleaned['season'].value_counts())

print("\nTime of Day Distribution:")
print(df_cleaned['time_of_day'].value_counts())

# Display a sample of the final dataset
print("\nSample of the final processed dataset:")
print(df_cleaned.head(30))

# ==========================================
# Investigate Outliers and Class Imbalance
# ==========================================

print("\n\n===== INVESTIGATING AGE OUTLIERS =====")
# Look at the cats with the highest ages
print("Cats with extremely high ages (> 15 years):")
old_cats = df_cleaned[df_cleaned['outcome_age_(days)'] > 5475]  # > 15 years
print(f"Number of very old cats: {len(old_cats)}")
print(old_cats[['outcome_age_(days)', 'breed1', 'outcome_type', 'sex_upon_outcome']].head(10))

# Show age distribution with a histogram
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['outcome_age_(days)'], bins=50)
plt.title('Distribution of Cat Ages (in days)')
plt.xlabel('Age in Days')
plt.ylabel('Count')
plt.axvline(x=5475, color='r', linestyle='--', label='15 years')
plt.legend()
plt.show()

# Check age statistics
print("\nAge statistics (in years):")
age_years = df_cleaned['outcome_age_(days)'] / 365
print(f"Mean: {age_years.mean():.2f}")
print(f"Median: {age_years.median():.2f}")
print(f"95th percentile: {age_years.quantile(0.95):.2f}")
print(f"99th percentile: {age_years.quantile(0.99):.2f}")

print("\n\n===== EXAMINING CLASS IMBALANCE =====")
# Show detailed class distribution
outcome_counts = df_cleaned['outcome_type'].value_counts()
outcome_percents = df_cleaned['outcome_type'].value_counts(normalize=True) * 100

print("\nDetailed class distribution:")
for outcome, count in outcome_counts.items():
    print(f"{outcome}: {count} ({outcome_percents[outcome]:.2f}%)")

# Visualize class imbalance
plt.figure(figsize=(10, 6))
sns.countplot(y='outcome_type', data=df_cleaned, order=outcome_counts.index)
plt.title('Distribution of Cat Outcome Types')
plt.tight_layout()
plt.show()

# Consider potential class groupings
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

# Show new distribution
combined_percents = combined_counts / combined_counts.sum() * 100
print("\nDistribution after combining rare classes:")
for outcome, count in combined_counts.items():
    print(f"{outcome}: {count} ({combined_percents[outcome]:.2f}%)")

# Create a new column with combined classes
df_cleaned['outcome_type_grouped'] = df_cleaned['outcome_type'].apply(
    lambda x: 'Other' if x in other_classes else x
)
df_cleaned['outcome_type_grouped'] = df_cleaned['outcome_type_grouped'].astype('category')

# Check the new distribution
print("\nVerifying the new grouped outcome distribution:")
grouped_counts = df_cleaned['outcome_type_grouped'].value_counts()
grouped_percents = df_cleaned['outcome_type_grouped'].value_counts(normalize=True) * 100
for outcome, count in grouped_counts.items():
    print(f"{outcome}: {count} ({grouped_percents[outcome]:.2f}%)")

# Visualize the new distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='outcome_type_grouped', data=df_cleaned, order=grouped_counts.index)
plt.title('Distribution of Cat Outcome Types (Grouped)')
plt.tight_layout()
plt.show()

# Save the cleaned and processed dataset to a CSV file
output_file = 'cleaned_cat_outcomes.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to {output_file}")