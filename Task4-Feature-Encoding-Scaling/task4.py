import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("cleaned_titanic.csv")

print("Original Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# =====================================
# Label Encoding
# =====================================

print("\nApplying Label Encoding on Sex")

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

print(df[['Sex']].head())

# =====================================
# One Hot Encoding
# =====================================

print("\nApplying One-Hot Encoding on Embarked")

df = pd.get_dummies(
    df,
    columns=['Embarked'],
    drop_first=True
)

print(df.head())

# =====================================
# Feature Scaling
# =====================================

features = ['Age', 'Fare']

# Standardization

standard_scaler = StandardScaler()

df_standardized = df.copy()

df_standardized[features] = standard_scaler.fit_transform(
    df_standardized[features]
)

# Normalization

minmax_scaler = MinMaxScaler()

df_normalized = df.copy()

df_normalized[features] = minmax_scaler.fit_transform(
    df_normalized[features]
)

# =====================================
# Comparison Plots
# =====================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['Age'], kde=True)
plt.title("Original Age")

plt.subplot(1,2,2)
sns.histplot(df_standardized['Age'], kde=True)
plt.title("Standardized Age")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['Fare'], kde=True)
plt.title("Original Fare")

plt.subplot(1,2,2)
sns.histplot(df_normalized['Fare'], kde=True)
plt.title("Normalized Fare")

plt.tight_layout()
plt.show()

# =====================================
# Save Dataset
# =====================================

df.to_csv(
    "encoded_scaled_titanic.csv",
    index=False
)

print("\nEncoding and Scaling Completed Successfully!")