from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
print(df.head())

print(df.describe())

print(df.isnull().sum())

#There are no null values in the dataset


print(df.shape)

print(df.info())

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
# heatmap that will show the correlation of features with one another.

main_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

for i in main_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="species", y=df[i], data=df)
    plt.title(f'Boxplot of {i} by Species')
    plt.show()

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for j in features:
    mean = df[j].mean()
    median = df[j].median()
    std_dev = df[j].std()

    # Z-score calculation
    z_scores = np.abs((df[j] - mean) / std_dev)
    outliers = (z_scores > 3).sum()  # Outliers if Z-score > 3

    print(f"{j}:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Median: {median:.4f}")
    print(f"  Standard Deviation: {std_dev:.4f}")
    print(f"  Number of Outliers (Z-score method): {outliers}\n")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuray:", accuracy)
print("classification report:",classification_report(y_test, y_pred))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


#CROSS VALIDATION
#as the model us working too well on the training data this mean that the model could be overfitted as well.

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)  # Change `X` and `y` to your full dataset
print('Cross-Validation Scores:', cv_scores)
print('Mean Cross-Validation Score:',cv_scores.mean())


#High scores of cross validation suggest that the moel is workung well on the unseen data.

