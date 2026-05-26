import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("titanic.csv")

print("Original Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handle Missing Values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']]).ravel()

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")