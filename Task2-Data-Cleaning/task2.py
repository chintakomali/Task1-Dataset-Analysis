import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv("titanic.csv")

print("Original Dataset Shape:", df.shape)

# Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handle Missing Values
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']]).ravel()

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin Column
df.drop('Cabin', axis=1, inplace=True)

# Verify
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save Cleaned Dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")