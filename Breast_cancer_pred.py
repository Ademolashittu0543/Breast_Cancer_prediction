#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('breast-cancer.csv')
df


# In[2]:


df.isna().sum()


# In[3]:


#  Check data types for object/category dtypes
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Categorical variables:")
print(categorical_cols)



# In[4]:


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("numeric variables:")
print(numeric_cols)


# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis_encoded'] = le.fit_transform(df['diagnosis'])
df=df.drop(columns=['diagnosis','id'],axis=1)
df


# In[6]:


unique_values = set(df['diagnosis_encoded'].unique())
unique_values


# In[7]:


for col in df:
    if (df[col] < 0 ).any():
        print(f"Negative values found in {col}: {(df[col] < 0).sum()}")
    else:
        print('No inconsistent column')


# In[8]:


x=df.drop(columns=['diagnosis_encoded'],axis=1)
y=df.diagnosis_encoded


# In[9]:


df.describe()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 20))
for i, col in enumerate(x.columns):
    plt.subplot(7, 5, i+1)
    sns.histplot(x[col], kde=True, color='blue')
    plt.title(col, fontsize=10)
    plt.tight_layout()
plt.suptitle('Feature Distributions', y=1.02)
plt.show()


# In[11]:


plt.figure(figsize=(15, 8))
sns.boxplot(data=x, orient='h', palette='Set2')
plt.title('Boxplots of Features')
plt.xlabel('Value')
plt.show()


# In[12]:


plt.figure(figsize=(15, 6))
sns.violinplot(data=x, inner='quartile', palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Violin Plots Showing Feature Distributions')
plt.show()


# In[13]:


import scipy.stats as stats

plt.figure(figsize=(15, 20))
for i, col in enumerate(x.columns):
    plt.subplot(7, 5, i+1)
    stats.probplot(x[col], dist="norm", plot=plt)
    plt.title(col, fontsize=8)
    plt.tight_layout()
plt.suptitle('Q-Q Plots (Normality Check)', y=1.02)
plt.show()


# In[14]:


import numpy as np
# Select 5 random features for visualization
features = np.random.choice(x.columns, 5, replace=False)
sns.pairplot(x[features], diag_kind='kde', corner=True)
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()


# In[15]:


plt.figure(figsize=(15, 20))
for i, col in enumerate(x.columns):
    plt.subplot(7, 5, i+1)
    sns.ecdfplot(x[col], color='purple')
    plt.title(col, fontsize=8)
    plt.tight_layout()
plt.suptitle('Cumulative Distribution Functions', y=1.02)
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='radius_mean', hue='diagnosis_encoded', kde=True, element='step', palette='viridis')
plt.title('Radius Mean Distribution by Diagnosis')
plt.show()


# In[17]:


corr_matrix = x.corr()
plt.figure(figsize=(20, 16))
# Filter correlations > 0.7 (adjust threshold)
mask = (abs(corr_matrix) < 0.7) & (corr_matrix != 1)
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, fmt=".2f",
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
plt.title('High Correlations Only (|r| ≥ 0.7)', pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[18]:


from scipy import stats
z_scores = np.abs(stats.zscore(df))
outliers = (z_scores > 3).sum(axis=0)
print("Outliers (>3σ) per feature:\n", outliers)


# In[19]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

print(f"Original rows: {df.shape[0]}, Cleaned rows: {df_cleaned.shape[0]}")


# In[20]:


print(df.dtypes)


# In[21]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
X_scaled


# In[22]:


# Initializing PCA
pca = PCA(n_components=None)
x_pca=pca.fit_transform(X_scaled)

#Analyzing explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid()
plt.show()

# Geting feature importance from PCA components
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i}' for i in range(1, len(x.columns)+1)],
    index=x.columns
)

# Identifying most influential features on top 5 PCs
top_n_components = 5
print(f"\nTop features per principal component (PC1-{top_n_components}):")
print(loadings.iloc[:, :top_n_components].abs().idxmax())

# Feature importance based on absolute loadings
feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': np.sum(np.abs(loadings.iloc[:, :top_n_components]), axis=1)
}).sort_values('Importance', ascending=False)

print("\nOverall feature importance from PCA:")
print(feature_importance.head(30))


# In[23]:


x_pca.shape


# In[24]:


from sklearn.model_selection import train_test_split, GridSearchCV
X_pca_train, X_pca_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)
X_pca_train.shape, X_pca_test.shape, y_train.shape, y_test.shape


# In[25]:


import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
    },
    'Lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'selection': ['cyclic', 'random']
        }
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params': {
            'alpha': [0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
}

scores = []

for model_name, mp in models.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_pca_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })



# In[26]:


df_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_score


# In[27]:


plt.figure(figsize=(12, 6))
sns.barplot(data=df_score, x='best_score', y='model', palette='viridis')
plt.title('Model Comparison (Score)')
plt.xlim(0, 1)
plt.show()


# In[28]:


knn=KNeighborsRegressor(n_neighbors=5,weights='distance')
knn.fit(X_pca_train,y_train)
knn.predict(X_pca_test)


# In[29]:


knn.score(X_pca_test,y_test)


# In[30]:


import shap
import numpy as np

background = shap.kmeans(X_pca_train, 10) 
explainer = shap.KernelExplainer(knn.predict, background)

shap_values = explainer.shap_values(X_pca_test[:100])
if isinstance(shap_values, list):  
    shap_values = np.array(shap_values)[0] 

num_features = X_pca_train.shape[1] 

top_n_components = 29
feature_names=loadings.iloc[:, :top_n_components].abs().idxmax()


shap.summary_plot(shap_values, X_pca_test[:100], feature_names=feature_names)

plt.tight_layout()
plt.show()


#     -->General Observations on Feature Influence in the Model Outcome<--
# 
# > Shape-related features (concave points, concavity, fractal dimension) are 
#  the most influential
# 
# • Malignant tumors tend to have irregular, concave, and complex boundaries.
# 
# • Features like concave points_mean and concavity_mean have the strongest 
#   impact on the model's prediction.
# 
# > Size-related features (radius, perimeter, area) also play a significant role
# 
# • Larger tumors are more likely to be malignant.
#  
# • Features like radius_worst, perimeter_worst, and area_worst contribute 
#   positively to malignancy classification.
# 
# > Texture and symmetry have moderate influence
# 
# • Malignant tumors tend to have more heterogeneous texture and asymmetry.
# 
# • Features like texture_worst and symmetry_worst contribute to the prediction 
#  but with less impact compared to shape and size-related features.
# 
# > Feature impact varies across data points
# 
# • Some features show a wide range of SHAP values, indicating that their 
#  influence depends on individual tumor characteristics.
# 
# • For example, concave points_mean has a consistent and strong positive 
#  impact, while symmetry_mean has a more scattered effect.
# 
# > Low feature values (blue) and high feature values (red) impact predictions 
#  differently
# 
# • Higher values of key features (e.g., concave points, radius, texture) 
#  increase the model’s probability of predicting malignancy.
# 
# • Lower values generally push predictions toward benign classification.
# 
# > Some features have minimal effect
# 
# • Features like fractal_dimension_worst and smoothness_se show SHAP values 
#   centered around zero, indicating they have little influence on the model’s 
#   decision.

# In[ ]:




