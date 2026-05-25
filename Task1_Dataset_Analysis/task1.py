import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DATASET DESCRIPTION =====")
print(df.describe())

print("\n===== SHAPE OF DATASET =====")
print(df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== NUMERICAL COLUMNS =====")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(numerical_cols)

print("\n===== CATEGORICAL COLUMNS =====")
categorical_cols = df.select_dtypes(include=['object']).columns
print(categorical_cols)

print("\n===== UNIQUE VALUES IN CATEGORICAL COLUMNS =====")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].unique())

print("\nNumerical Columns:")
print(df.select_dtypes(include=['int64','float64']).columns)

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))