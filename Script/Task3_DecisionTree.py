------------#Step1: Import Libraries-----------
# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")

------------#Step2: Load Dataset-----------
df = pd.read_csv(r"C:\Users\admin\Documents\Prodigy Infotech\Prodigy_DS_03\bank+marketing\bank\bank-full.csv",
    sep=';')

df.head()

------------#Step3: Data Understanding-----------
------------#Dataset Info------------
df.info()

------------#Statistical Summary---------------
df.describe()

------------#Check Missing Values------------
df.isnull().sum()

------------#Step4: Basic EDA---------------
------------#Target distribution------------
sns.countplot(x='y', data=df)
plt.title("Purchase Distribution")
plt.show()

------------#Purchase by Job-----------
plt.figure(figsize=(10,5))
sns.countplot(x='job', hue='y', data=df)
plt.xticks(rotation=45)
plt.show()

------------#Purchase by Marital Status----------
sns.countplot(x='marital', hue='y', data=df)
plt.show()

------------#Step5: Data Preprocessing------------
------------#Convert categorical → numeric--------
df_encoded = pd.get_dummies(df, drop_first=True)

------------#Step6: Feature & Target Split-----------
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']

------------#Step7: Train Test Split---------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

------------#Step8: Decision Tree Model-------------
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

------------#Step9: Prediction-----------
y_pred = model.predict(X_test)

------------#Step10: Evaluation------------
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

-----------#Step11: Decision Tree Visualization-----------
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No','Yes'])
plt.title("Decision Tree Visualization")
plt.show()

------------#Step12: Feature Importance-----------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).plot(kind='bar', figsize=(10,5))
plt.title("Feature Importance")
plt.show()
