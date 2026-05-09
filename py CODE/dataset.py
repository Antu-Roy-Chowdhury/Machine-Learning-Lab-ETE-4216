# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from seaborn import heatmap

# %%
df=pd.read_csv("data/car_evaluation.tsv", sep='\t')
df
x_col = len(df.columns) - 1


# %%
#list all columns names in df as tables raw with index sn numbrt of null values in each column
for i in range(len(df.columns)):
    print(f"{i}: {df.columns[i]} - {df.iloc[:, i].isnull().sum()}")

# %%
X = df.iloc[:, 0:x_col].values
y = df.iloc[:, x_col].values


# %%
# count of each class in y
class_counts = np.bincount(y)
for i, count in enumerate(class_counts):
    print(f'Class {i}: {count} samples')

# %%
#plot data features list and count on df
for i in range(x_col):
    plt.subplot(2, 3, i + 1)
    plt.hist(X[:, i], bins=20)
    plt.xlabel(df.columns[i])
    plt.ylabel('Count')
    plt.title(f'Distribution of {df.columns[i]}')
plt.tight_layout()
plt.savefig('Figure/feature_distributions.pdf')
plt.show()

# %%
x_col= len(df.columns[:-1])

# %%
print(df.isnull().sum())

# %%

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('Figure/correlation_matrix.pdf')
plt.show()

# %%
# Visualize outliers using a boxplot
sns.boxplot(data=df,palette="Blues")
plt.title('Boxplot for Outlier Detection')
plt.savefig('Figure/boxplot_outliers.pdf')
plt.show()

# You can use Z-score or IQR to remove extreme outliers if necessary

# %%
X = df.iloc[:, 0:x_col].values
print(X)


# %%
y = df.iloc[:, -1].values
print(y)

# %%


# %%
# plot data varition on catagory of y
sns.countplot(x=y)
plt.title('Distribution of Target Variable')    
plt.xlabel('Car Evaluation Category')
plt.ylabel('Count')
plt.savefig('Figure/target_variable_distribution.pdf')
plt.show()


