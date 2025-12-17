
# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 2. Load Dataset
df = pd.read_csv("../data/Titanic-Dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())
print("\nDataset Info:")
print(df.info())


# 3. Data Preprocessing


# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())            
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  
df['Fare'] = df['Fare'].fillna(df['Fare'].median())


# 4. Encoding Categorical Variables

# Encode Sex (binary encoding)
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked (one-hot encoding)
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)

# Encode Pclass (one-hot encoding)
pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
df = pd.concat([df, pclass_dummies], axis=1)


# 5. Feature Engineering


# FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# IsAlone
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)

# Age binning
bins = [0, 12, 19, 59, 100]
labels = ['Child', 'Teen', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Encode AgeGroup
age_dummies = pd.get_dummies(df['AgeGroup'], prefix='Age')
df = pd.concat([df, age_dummies], axis=1)


# 6. Exploratory Data Analysis (EDA)


# Survival by Gender
df.groupby('Sex')['Survived'].mean().plot(kind='bar', title='Survival Rate by Gender')
plt.ylabel("Survival Rate")
plt.show()

# Survival by Passenger Class
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', stacked=True)
plt.title("Survival by Passenger Class")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

# Survival by Family Size
df.groupby('FamilySize')['Survived'].mean().plot(marker='o')
plt.title("Survival Rate by Family Size")
plt.ylabel("Survival Rate")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['Survived','Age','Fare','FamilySize','IsAlone','Sex_encoded']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 7. Model Building


# Feature selection
features = [
    'Sex_encoded', 'Age', 'Fare', 'FamilySize', 'IsAlone',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'Pclass_1', 'Pclass_2', 'Pclass_3',
    'Age_Child', 'Age_Teen', 'Age_Adult', 'Age_Senior'
]

X = df[features]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 8. Logistic Regression Model

lr_model = LogisticRegression(max_iter=300)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))


# 9. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))


# 10. Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nProject completed successfully.")
