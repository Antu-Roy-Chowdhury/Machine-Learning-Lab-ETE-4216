# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve

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
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(X[:, 0:x_col])
X[:, 0:x_col] = imputer.transform(X[:, 0:x_col])

# %%
print(X)

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
accuracy = {}
best_acc = 0
best_pred = 0
for i in range(1, 20):
    model = DecisionTreeClassifier(random_state=42, max_depth=i, criterion='entropy')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy[i] = accuracy_score(y_test, y_pred)
    if accuracy[i] > best_acc:
        best_acc = accuracy[i]
        best_pred = y_pred
max_depth = max(accuracy, key=accuracy.get)
best_acc = accuracy[max_depth]

plt.plot(list(accuracy.keys()), list(accuracy.values()), marker='o', label='Decision Tree')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.legend()
plt.scatter(max_depth, best_acc, color='red', zorder=5)
plt.annotate(f'max_depth: {max_depth}\nAccuracy: {best_acc:.4f}', xy=(max_depth, best_acc), xytext=(max_depth - 1.5, best_acc - 0.09), arrowprops=dict(arrowstyle='->', color='red'))

plt.savefig('Figure/decision_tree_accuracy.pdf')
plt.show()

# %%
# plot tree on best max_depth
from sklearn.tree import plot_tree
best_model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, criterion='entropy')
best_model.fit(X_train_scaled, y_train)
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    filled=True,
    feature_names=df.columns[:-1].astype(str),
    class_names=[str(c) for c in le.classes_]
)
plt.savefig('Figure/decision_tree_plot.pdf')
plt.show()

# %%
feature_importance = pd.Series(best_model.feature_importances_, index=df.columns[:-1])
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance for Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig('Figure/decision_tree_feature_importance.pdf')
plt.show()

# %%
cm = confusion_matrix(y_test, best_pred)

print("Accuracy:", accuracy_score(y_test, best_pred))

report = classification_report(y_test, best_pred)
print(report)

# %%
cm

# %%
plt.figure(figsize=(8, 6))
plt.title("Confusion Matrix for Decision Tree")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Figure/decision_tree_confusion_matrix.pdf')
plt.show()

# %% [markdown]
# # Random Forest

# %%
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred2 = model.predict(X_test_scaled)

# %%
cm2 = confusion_matrix(y_test, y_pred2)
print("Accuracy:", accuracy_score(y_test, y_pred2))
report2 = classification_report(y_test, y_pred2)
print(report2)

# %%
cm2

# %%
rf_feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1])
rf_feature_importance = rf_feature_importance.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=rf_feature_importance.values, y=rf_feature_importance.index)
plt.title('Feature Importance for Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig('Figure/random_forest_feature_importance.pdf')
plt.show()

# %%
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix for Random Forest')
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Figure/random_forest_confusion_matrix.pdf')
plt.show()

# %%

report_dt = classification_report(y_test, best_pred, output_dict=True)
report_rf = classification_report(y_test, y_pred2, output_dict=True)
comparison = pd.DataFrame({
    'Decision Tree': [report_dt['0']['precision'], report_dt['0']['recall'], report_dt['0']['f1-score'], report_dt['accuracy']],
    'Random Forest': [report_rf['0']['precision'], report_rf['0']['recall'], report_rf['0']['f1-score'], report_rf['accuracy']]
}, index=['Precision', 'Recall', 'F1-Score', 'Accuracy'])

comparison = comparison.round(4)
comparison



