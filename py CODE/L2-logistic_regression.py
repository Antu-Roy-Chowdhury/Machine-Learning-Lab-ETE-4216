# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from seaborn import heatmap


# %%
df=pd.read_csv("data/car_evaluation.tsv", sep='\t')
df

# %%
x_col= len(df.columns[:-1])

# %%
X = df.iloc[:, 0:x_col].values
print(X)


# %%
y = df.iloc[:, -1].values
print(y)

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
from sklearn.preprocessing import StandardScaler
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

# %%
model=LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

# %%
y_pred = model.predict(X_test_scaled)

# %%
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='accuracy'
)

# %%

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Learning Curve")

plt.legend()
plt.savefig('Figure/logistic_learning_curve.pdf')
plt.show()

# %%

cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
# plot cm with values in every cell
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Figure/logistic_regression_confusion_matrix.pdf')
plt.show()

# %%

report = classification_report(y_test, y_pred)
print(report)



# %%


# %% [markdown]
# # KNN

# %%
accuracy = {}

best_acc = 0
best_model = None
best_pred = None
best_k = 0

for k in range(1, 11):
    model2=KNeighborsClassifier(n_neighbors=k)
    model2.fit(X_train_scaled, y_train)
    y_pred2 = model2.predict(X_test_scaled)
    accuracy[k] = np.mean(y_pred2 == y_test)
    if accuracy[k] > best_acc:
        best_acc = accuracy[k]
        best_model = model2
        best_pred = y_pred2
        best_k = k

plt.plot(list(accuracy.keys()), list(accuracy.values()), marker='o')
plt.title('KNN Accuracy for Different n_neighbors')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.xticks(range(1, 11))
plt.scatter(best_k, best_acc, color='red', zorder=5)
plt.annotate(f'Best n_neighbors: {best_k}\nAccuracy: {best_acc:.4f}', xy=(best_k, best_acc), xytext=(best_k + 0.05, best_acc - 0.04), arrowprops=dict(arrowstyle='->', color='red'))
plt.savefig('Figure/knn_accuracy.pdf')
plt.show()

print(f"Best n_neighbors: {best_k} with accuracy: {best_acc:.4f}")

# %%

report = classification_report(y_test, best_pred)
print(report)


# %%

cm = confusion_matrix(y_test, best_pred)
# plot cm with values in every cell
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Figure/kn_confusion_matrix.pdf')
plt.show()

# %%



