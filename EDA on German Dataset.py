#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["checking_account_status", "duration_month", "credit_history", "purpose", "credit_amount",
           "savings_account", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
           "present_residence_since", "property", "age", "other_installment_plans", "housing", 
           "existing_credits", "job", "liable_people", "telephone", "foreign_worker", "class"]
data = pd.read_csv(url, sep=" ", names=columns)

# Map class labels to meaningful names
class_mapping = {1: 'Good', 2: 'Bad'}
data['class'] = data['class'].map(class_mapping)

# Visualize some categorical attributes
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.countplot(x='checking_account_status', data=data)
plt.title("Checking Account Status")

plt.subplot(2, 2, 2)
sns.countplot(x='credit_history', data=data)
plt.title("Credit History")

plt.subplot(2, 2, 3)
sns.countplot(x='purpose', data=data)
plt.title("Purpose")

plt.subplot(2, 2, 4)
sns.countplot(x='savings_account', data=data)
plt.title("Savings Account")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[18]:


# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Display summary statistics for numerical attributes
print("\nSummary Statistics for Numerical Attributes:")
print(data.describe())

# Display summary statistics for categorical attributes
print("\nSummary Statistics for Categorical Attributes:")
print(data.describe(include=['object']))


# In[8]:


# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=data)
plt.title("Distribution of Class Labels")
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[15]:


# Using RandomForestClassifier to get feature importances
from sklearn.ensemble import RandomForestClassifier

# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data.drop('class', axis=1))

# Split the data into features (X) and target variable (y)
X = data_encoded
y = data['class']

# Train the Random Forest classifier
rf.fit(X, y)

# Get feature importances
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
print("Feature Importances:")
print(feature_importances)


# In[24]:


# Select numerical columns for outlier detection
numerical_cols = data.select_dtypes(include=[np.number])

# Define threshold for Z-score (e.g., 3 or -3)
threshold = 3

# Calculate Z-score for each data point in numerical columns
z_scores = ((numerical_cols - numerical_cols.mean()) / numerical_cols.std()).abs()

# Identify outliers based on Z-score exceeding the threshold
outliers_count = (z_scores > threshold).sum()

    
    # Print the count of outliers for each numerical column
print("Count of outliers:")
print(outliers_count)


# In[ ]:




