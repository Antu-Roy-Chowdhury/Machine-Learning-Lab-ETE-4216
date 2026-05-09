# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv("data/car_evaluation.tsv", sep='\t')
df

# %%
x_col = len(df.columns[:-1])

# %%
X = df.iloc[:, 0:x_col].values
print(X)

# %%
y = df.iloc[:, -1].values
print(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## SVM

# %%
model = SVC(kernel='rbf', degree=4, gamma='scale', C=1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# %%
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred)
print(report)

# %%
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Figure/svc_confusion_matrix.pdf')
plt.show()

# %%
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("SVC Learning Curve")

plt.legend()
plt.savefig('Figure/svc_learning_curve.pdf')
plt.show()

# %% [markdown]
# ## Naive Bayes

# %%
model_NB = MultinomialNB()
model_NB.fit(X_train, y_train)
y_pred2 = model_NB.predict(X_test)

# %%
cm2 = confusion_matrix(y_test, y_pred2)
print("Accuracy:", accuracy_score(y_test, y_pred2))
report2 = classification_report(y_test, y_pred2)
print(report2)

# %%
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix for Naive Bayes')
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Figure/nb_confusion_matrix.pdf')
plt.show()

# %%
train_sizes, train_scores, test_scores = learning_curve(
    model_NB,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Naive Bayes Learning Curve")

plt.legend()
plt.savefig('Figure/nb_learning_curve.pdf')
plt.show()


