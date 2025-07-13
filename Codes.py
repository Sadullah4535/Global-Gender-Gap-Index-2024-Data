#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[198]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define subindex columns
subindexes = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']

# Select top 10 and bottom 10 countries based on GGGI Score
top_10 = df.sort_values('GGGI Score', ascending=False).head(10)
bottom_10 = df.sort_values('GGGI Score', ascending=True).head(10)

# Melt the data for plotting
top_10_melted = top_10.melt(id_vars='Country', value_vars=subindexes, var_name='Subindex', value_name='Score')
bottom_10_melted = bottom_10.melt(id_vars='Country', value_vars=subindexes, var_name='Subindex', value_name='Score')

# -------------------
# Plot: Top 10 Countries
# -------------------
plt.figure(figsize=(14, 12))
top_plot = sns.barplot(data=top_10_melted, x='Subindex', y='Score', hue='Country')
plt.title('Top 10 Countries by GGGI Score - Subindex Comparison')
plt.ylabel('Score')
plt.xlabel('')
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)

# Add rotated labels
for container in top_plot.containers:
    top_plot.bar_label(container, fmt='%.3f', fontsize=8, label_type='edge', padding=2, rotation=90)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------
# Plot: Bottom 10 Countries
# -------------------
plt.figure(figsize=(14, 12))
bottom_plot = sns.barplot(data=bottom_10_melted, x='Subindex', y='Score', hue='Country')
plt.title('Bottom 10 Countries by GGGI Score - Subindex Comparison')
plt.ylabel('Score')
plt.xlabel('Subindex')
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)

# Add rotated labels
for container in bottom_plot.containers:
    bottom_plot.bar_label(container, fmt='%.3f', fontsize=8, label_type='edge', padding=2, rotation=90)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Alt endeks sütunları
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']

# GGGI skoruna göre ilk 10 ve son 10 ülkeyi seç
top_10 = df.sort_values('GGGI Score', ascending=False).head(10)
bottom_10 = df.sort_values('GGGI Score').head(10)

# Plot için melt işlemi
top_10_melted = top_10.melt(id_vars='Country', value_vars=features, var_name='Subindex', value_name='Score')
bottom_10_melted = bottom_10.melt(id_vars='Country', value_vars=features, var_name='Subindex', value_name='Score')

# Grafik düzeni: 2 satır 1 sütun
fig, axes = plt.subplots(2, 1, figsize=(14, 16), sharex=True)

# İlk 10 ülke plotu
top_plot = sns.barplot(data=top_10_melted, x='Subindex', y='Score', hue='Country', ax=axes[0])
axes[0].set_title('Top 10 Countries by GGGI Score - Subindex Comparison')
axes[0].set_ylabel('Score')
axes[0].set_xlabel('')
axes[0].axhline(1, color='gray', linestyle='--', linewidth=0.8)

# Veri etiketleri (90 derece döndürülmüş)
for container in top_plot.containers:
    top_plot.bar_label(container, fmt='%.3f', fontsize=8, label_type='edge', padding=2, rotation=90)

axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Country')

# Son 10 ülke plotu
bottom_plot = sns.barplot(data=bottom_10_melted, x='Subindex', y='Score', hue='Country', ax=axes[1])
axes[1].set_title('Bottom 10 Countries by GGGI Score - Subindex Comparison')
axes[1].set_ylabel('Score')
axes[1].set_xlabel('Subindex')
axes[1].axhline(1, color='gray', linestyle='--', linewidth=0.8)

# Veri etiketleri (90 derece döndürülmüş)
for container in bottom_plot.containers:
    bottom_plot.bar_label(container, fmt='%.3f', fontsize=8, label_type='edge', padding=2, rotation=90)

axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Country')

# Son ayarlamalar
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:





# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data with correct encoding and separator
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Correlation matrix for selected Gender Gap Index components
corr = df[['GGGI Score', 
           'Economic Participation and Opportunity', 
           'Educational Attainment', 
           'Health and Survival', 
           'Political Empowerment']].corr()

# Visualization of the correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='cool')
plt.title("Gender Gap Index - Correlation Matrix")
plt.show()

# Print correlation matrix
print(corr)



# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
import numpy as np

# 0. Read the data
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# 1. Visualize the distribution of the dependent variable and perform the Shapiro-Wilk test
stat, p = shapiro(df['GGGI Score'])
print(f'Shapiro-Wilk Test: Statistic={stat:.3f}, p-value={p:.3f}')
if p > 0.05:
    print("The GGGI Score variable is normally distributed (null hypothesis cannot be rejected).")
else:
    print("The GGGI Score variable is not normally distributed (null hypothesis rejected). Consider a transformation.")

# 2. Build the multiple linear regression model
X = df[['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']]
y = df['GGGI Score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# 3. Residual analysis
fitted_vals = model.fittedvalues
residuals = model.resid

# 4. Plot all diagnostics in a single figure (4 rows, 1 column)
fig, axes = plt.subplots(4, 1, figsize=(8, 16))  # 4 rows, 1 column

# 4.1 Distribution of GGGI Score
sns.histplot(df['GGGI Score'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of GGGI Score')

# 4.2 Distribution of Residuals
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title('Distribution of Residuals')

# 4.3 Q-Q Plot of Residuals
qq = sm.qqplot(residuals, line='s', ax=axes[2])
axes[2].set_title('Q-Q Plot of Residuals')

# 4.4 Residuals vs Fitted Values
axes[3].scatter(fitted_vals, residuals)
axes[3].axhline(0, color='red', linestyle='--')
axes[3].set_xlabel('Fitted Values')
axes[3].set_ylabel('Residuals')
axes[3].set_title('Residuals vs Fitted Values')

plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
import numpy as np

# 0. Veriyi oku
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# 1. Bağımlı değişkenin dağılımını görselleştir ve Shapiro-Wilk testi uygula
plt.figure(figsize=(8,4))
sns.histplot(df['GGGI Score'], kde=True)
plt.title('Distribution of GGGI Score')
plt.tight_layout()
plt.show()

stat, p = shapiro(df['GGGI Score'])
print(f'Shapiro-Wilk Test: Statistic={stat:.3f}, p-value={p:.3f}')
if p > 0.05:
    print("GGGI Score değişkeni normal dağılıma sahip (hipotez reddedilemez).")
else:
    print("GGGI Score değişkeni normal dağılmıyor (hipotez reddedildi). Dönüşüm düşünülebilir.")

# 2. Çoklu regresyon modeli kurma
X = df[['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']]
y = df['GGGI Score']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# 3. Residual analizleri

# Tahmin edilen değerler ve residuals
fitted_vals = model.fittedvalues
residuals = model.resid

# 3.1 Residual histogramı
plt.figure(figsize=(8,4))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.tight_layout()
plt.show()

# 3.2 Q-Q plot
plt.figure(figsize=(6,6))
sm.qqplot(residuals, line='s', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.show()

# 3.3 Residuals vs Fitted
plt.figure(figsize=(8,4))
plt.scatter(fitted_vals, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Veri yükleme
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Özellikler ve hedef değişken
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']
X = df[features]
y = df['GGGI Score']

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest Modeli
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Performans değerlendirme
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}')

# 2. XGBoost Modeli
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Performans değerlendirme
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost - MSE: {mse_xgb:.4f}, R2: {r2_xgb:.4f}')

# Değişken önemlerinin görselleştirilmesi
def plot_feature_importances(model, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.title(f'{model_name} Feature Importances')
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices])
    plt.ylabel('Importance')
    plt.show()

plot_feature_importances(rf, "Random Forest")
plot_feature_importances(xgb, "XGBoost")


# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Veri yükleme
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Özellikler ve hedef değişken
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']
X = df[features]
y = df['GGGI Score']

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımla
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

# Sonuçlar için boş listeler
results = []

def plot_feature_importances(model, model_name):
    # Bazı modeller feature_importances_ özelliğine sahip değil, kontrol et
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,5))
        plt.title(f'{model_name} Feature Importances')
        bars = plt.bar(range(len(features)), importances[indices], align='center')
        plt.xticks(range(len(features)), [features[i] for i in indices])
        plt.ylabel('Importance')
        # Barların üstüne değerleri yaz
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')
        plt.show()
    else:
        print(f"{model_name} modelinde feature_importances_ özelliği bulunmamaktadır.")

# Tüm modelleri çalıştır, değerlendir ve görselleştir
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MSE': mse, 'R2': r2})
    print(f'{name} - MSE: {mse:.4f}, R2: {r2:.4f}')
    plot_feature_importances(model, name)

# Performans karşılaştırma grafiği
results_df = pd.DataFrame(results)
plt.figure(figsize=(10,6))
bar_width = 0.35
index = np.arange(len(results_df))

plt.bar(index, results_df['R2'], bar_width, label='R2 Score', color='royalblue')
plt.bar(index + bar_width, -results_df['MSE'], bar_width, label='-MSE (negatif)', color='salmon')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performans Karşılaştırması (R2 ve -MSE)')
plt.xticks(index + bar_width / 2, results_df['Model'])
plt.legend()

# Barların üstüne değerleri yaz
for i in range(len(results_df)):
    plt.text(i, results_df['R2'][i] + 0.02, f"{results_df['R2'][i]:.3f}", ha='center', color='black')
    plt.text(i + bar_width, -results_df['MSE'][i] + 0.02, f"{results_df['MSE'][i]:.3f}", ha='center', color='black')

plt.ylim(min(-results_df['MSE']) - 0.1, 1.1)
plt.show()


# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data (latin1 encoding, semicolon separator)
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Features and target variable
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']
X = df[features]

# Target variable (assumed continuous)
y = df['GGGI Score'].astype(float)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

# Colors for plots
colors = {
    "Random Forest": 'royalblue',
    "XGBoost": 'darkorange',
    "Gradient Boosting": 'seagreen',
    "SVR": 'purple'
}

def plot_feature_importances(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,5))
        plt.title(f'{model_name} Feature Importances')
        bars = plt.bar(range(len(features)), importances[indices], align='center', color=colors[model_name])
        plt.xticks(range(len(features)), [features[i] for i in indices])
        plt.ylabel('Importance')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')
        plt.show()
    else:
        print(f"{model_name} does not have feature_importances_ attribute.")

# Train models, evaluate and plot feature importances
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - MSE: {mse:.4f}, R2: {r2:.4f}')
    
    plot_feature_importances(model, name)


# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data (latin1 encoding, semicolon separator)
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Features and target variable
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']
X = df[features]

# Target variable (continuous)
y = df['GGGI Score'].astype(float)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

# Colors for plots
colors = {
    "Random Forest": 'royalblue',
    "XGBoost": 'darkorange',
    "Gradient Boosting": 'seagreen',
    "SVR": 'purple'
}

# Prepare subplots (4 models, 1 column, 4 rows)
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 20))
fig.tight_layout(pad=5.0)

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R2: {r2:.4f}')
    
    # Plot feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        ax = axs[i]
        ax.bar(range(len(features)), importances[indices], color=colors[name])
        ax.set_title(f'{name} Feature Importances\nMSE: {mse:.4f}, R2: {r2:.4f}')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([features[j] for j in indices], rotation=45, ha='right')
        ax.set_ylabel('Importance')
        
        # Add value labels on bars
        for j in range(len(features)):
            height = importances[indices][j]
            ax.text(j, height + 0.01, f'{height:.3f}', ha='center', va='bottom')
    else:
        # If no feature_importances_, just show metrics in the subplot
        axs[i].text(0.5, 0.5, f"{name} does not have feature_importances_\nMSE: {mse:.4f}, R2: {r2:.4f}", 
                    ha='center', va='center', fontsize=12)
        axs[i].axis('off')

plt.show()


# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data (latin1 encoding, semicolon separator)
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Features and target variable
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']
X = df[features]

# Target variable (continuous)
y = df['GGGI Score'].astype(float)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to plot (only those with feature_importances_)
models_to_plot = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Colors for plots
colors = {
    "Random Forest": 'royalblue',
    "XGBoost": 'darkorange',
    "Gradient Boosting": 'seagreen'
}

# Prepare subplots (1 row, 3 columns)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.tight_layout(pad=5.0)

for i, (name, model) in enumerate(models_to_plot.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R2: {r2:.4f}')
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax = axs[i]
    ax.bar(range(len(features)), importances[indices], color=colors[name])
    ax.set_title(f'{name} Feature Importances\nMSE: {mse:.4f}, R2: {r2:.4f}')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[j] for j in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance')
    
    # Add value labels on bars
    for j in range(len(features)):
        height = importances[indices][j]
        ax.text(j, height + 0.01, f'{height:.3f}', ha='center', va='bottom')

plt.show()

print(models_to_plot)


# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load data (latin1 encoding, semicolon separator)
df = pd.read_csv('Data.csv', encoding='latin1', sep=';')

# Features and target variable
features = ['Economic Participation and Opportunity', 'Educational Attainment', 'Health and Survival', 'Political Empowerment']
X = df[features]

# Target variable (continuous)
y = df['GGGI Score'].astype(float)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to plot (only those with feature_importances_)
models_to_plot = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Colors for plots
colors = {
    "Random Forest": 'royalblue',
    "XGBoost": 'darkorange',
    "Gradient Boosting": 'seagreen'
}

# To store results for printing later
results = []

# Prepare subplots (1 row, 3 columns)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
fig.tight_layout(pad=5.0)

for i, (name, model) in enumerate(models_to_plot.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name} - MSE: {mse:.4f}, R2: {r2:.4f}')
    results.append({'Model': name, 'MSE': mse, 'R2': r2})
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax = axs[i]
    ax.bar(range(len(features)), importances[indices], color=colors[name])
    ax.set_title(f'{name} Feature Importances\nMSE: {mse:.4f}, R2: {r2:.4f}')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[j] for j in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance')
    
    # Add value labels on bars
    for j in range(len(features)):
        height = importances[indices][j]
        ax.text(j, height + 0.01, f'{height:.3f}', ha='center', va='bottom')

plt.show()

# Print summary table of results
print("\nSummary of Model Performances:")
for res in results:
    print(f"{res['Model']}: MSE = {res['MSE']:.4f}, R2 = {res['R2']:.4f}")


# In[ ]:





# In[54]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS', color=color)
ax1.plot(k_range, wcss, marker='o', color=color, label='WCSS')
for i, value in enumerate(wcss):
    ax1.text(k_range[i], value * 1.01, f'{value:.1f}', ha='center', fontsize=8, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Aynı x-ekseni için ikinci y-ekseni
color = 'tab:green'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, marker='s', color=color, label='Silhouette Score')
for i, value in enumerate(silhouette_scores):
    ax2.text(k_range[i], value - 0.03, f'{value:.2f}', ha='center', fontsize=8, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# K=4 için kırmızı kesikli çizgi
ax1.axvline(x=4, color='red', linestyle='--')

plt.title('WCSS and Silhouette Score by Number of Clusters (k)')
fig.tight_layout()
plt.grid(True)
plt.show()


# In[ ]:





# In[55]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS', color=color)
ax1.plot(k_range, wcss, marker='o', color=color, label='WCSS')
for i, value in enumerate(wcss):
    ax1.text(k_range[i], value + (max(wcss)*0.02), f'{value:.1f}', ha='center', fontsize=8, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Aynı x-ekseni için ikinci y-ekseni
color = 'tab:green'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, marker='s', color=color, label='Silhouette Score')
for i, value in enumerate(silhouette_scores):
    ax2.text(k_range[i], value - 0.03, f'{value:.2f}', ha='center', fontsize=8, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# K=4 için kırmızı kesikli çizgi
ax1.axvline(x=4, color='red', linestyle='--')

plt.title('WCSS and Silhouette Score by Number of Clusters (k)')
fig.tight_layout()
plt.grid(True)
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Index hatasını önlemek için sınır belirle
n = min(len(k_range), len(wcss), len(silhouette_scores), 15)

print("k\tWCSS\t\tSilhouette Score")
print("-" * 30)
for i in range(n):
    print(f"{k_range[i]}\t{wcss[i]:.2f}\t\t{silhouette_scores[i]:.4f}")

plt.figure(figsize=(10,6))
fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS', color=color)
ax1.plot(k_range, wcss, marker='o', color=color, label='WCSS')
for i, value in enumerate(wcss):
    ax1.text(k_range[i] + 0.15, value + max(wcss)*0.01, f'{value:.1f}', ha='left', va='bottom', fontsize=8, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Aynı x-ekseni için ikinci y-ekseni
color = 'tab:green'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, marker='s', color=color, label='Silhouette Score')
for i, value in enumerate(silhouette_scores):
    ax2.text(k_range[i] + 0.15, value - 0.01, f'{value:.2f}', ha='left', va='top', fontsize=8, color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.axvline(x=7, color='red', linestyle='--')

plt.title('WCSS and Silhouette Score by Number of Clusters (k)')
fig.tight_layout()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[106]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.patches as mpatches
import numpy as np

# Özellik sütunları
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]
y_true = df['True_Label']  # Gerçek sınıflar burada olmalı (örn. kıta, gelişmişlik seviyesi vs.)

# KMeans kümeleme
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# PCA ile 2B'ye indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Yeni figür: scatter plot + ülke listesi sağda
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 2]})
cmap = plt.cm.viridis

# --- Sol: PCA scatter ---
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=70)
ax1.set_title('K-Means Clustering Visualization with PCA (k=7)', fontsize=14)
ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
ax1.grid(True)
plt.colorbar(scatter, ax=ax1, label='GGGI Score')

# --- ARI ve NMI metrikleri ---
ari = adjusted_rand_score(y_true.astype(int), y_pred)
nmi = normalized_mutual_info_score(y_true.astype(int), y_pred)
textstr = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"
ax1.text(x=min(X_pca[:, 0]), y=max(X_pca[:, 1]) * 1.1, s=textstr, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7))

# --- Sağ: Küme etiketleri ve ülkeler ---
ax2.axis('off')
cluster_labels = range(7)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

y_position = 1.0
for label in cluster_labels:
    countries = cluster_countries[label]
    color = cmap(label / max(cluster_labels))
    country_text = f"Cluster {label} ({len(countries)} countries):\n" + ", ".join(countries)
    
    ax2.text(0, y_position, country_text, fontsize=10, verticalalignment='top', color=color, wrap=True)
    y_position -= 0.05 + 0.015 * len(countries)  # Dinamik aralık

# --- Konsola da yazdır ---
print("CLUSTER ÜLKE LİSTESİ:\n" + "="*60)
for label in cluster_labels:
    countries = cluster_countries[label]
    print(f"Cluster {label} ({len(countries)} countries):")
    print(", ".join(countries))
    print("-" * 60)

plt.tight_layout()
plt.show()


# In[104]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import pandas as pd

# --- Örnek: df yükleme (kendi verinizi buraya yükleyin) ---
# df = pd.read_csv("your_data.csv")

# --- Özellikleri ve etiketleri tanımla ---
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]
y_true = df['GGGI Score']  # Gerçek etiket sütununu burada tanımlayın

# --- KMeans kümeleme ---
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# --- PCA ile 2B'ye indirgeme ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# --- Kümeleri 2D olarak çiz ---
plt.figure(figsize=(12, 8))
cmap = plt.cm.viridis
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=70)
plt.title('K-Means Clustering Visualization with PCA (k=7)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='GGGI Score')
plt.grid(True)

# --- ARI ve NMI hesapla ---
ari = adjusted_rand_score(y_true.astype(int), y_pred)
nmi = normalized_mutual_info_score(y_true.astype(int), y_pred)
textstr = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"
plt.text(x=min(X_pca[:, 0]), y=max(X_pca[:, 1]) * 1.4, s=textstr, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7))
plt.ylim(top=max(X_pca[:, 1]) * 1.1)

plt.show()

# --- Küme etiketlerine göre ülke listesi oluştur ---
cluster_labels = range(7)
cluster_countries = {
    label: df.loc[y_pred == label, 'Country'].tolist()
    for label in cluster_labels
}

# --- Ülkeleri ve küme renklerini gösteren yeni görsel ---
fig, ax = plt.subplots(figsize=(6, 12))
ax.axis('off')

text = ""
patches = []

for label in cluster_labels:
    countries = cluster_countries[label]
    color = cmap(label / (max(cluster_labels)))  # Normalize label for color
    patches.append(mpatches.Patch(color=color, label=f'Cluster {label}'))
    
    country_list = ", ".join(countries)
    text += f"Cluster {label} ({len(countries)} countries):\n"
    text += country_list + "\n\n"

# Küme renkleri için legend
ax.legend(handles=patches, title="Clusters", loc='upper right', fontsize=10)

# Ülke listesini yazdır
ax.text(0, 1, text, fontsize=10, verticalalignment='top', wrap=True)

plt.show()


# In[100]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.patches as mpatches
import numpy as np

features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 8))
cmap = plt.cm.viridis
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=70)
plt.title('K-Means Clustering Visualization with PCA (k=7)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)

ari = adjusted_rand_score(y_true.astype(int), y_pred)
nmi = normalized_mutual_info_score(y_true.astype(int), y_pred)
textstr = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"
plt.text(x=min(X_pca[:,0]), y=max(X_pca[:,1]) * 1.4, s=textstr, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7))
plt.ylim(top=max(X_pca[:,1]) * 1.1)

# Cluster countries dictionary
cluster_labels = range(7)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

# Create a new figure for the cluster country list with colors
fig, ax = plt.subplots(figsize=(5, 10))
ax.axis('off')

text = ""
patches = []
for label in cluster_labels:
    countries = cluster_countries[label]
    color = cmap(label / (max(cluster_labels)))  # Normalize label for cmap
    patches.append(mpatches.Patch(color=color, label=f'Cluster {label}'))
    
    countries_str = ", ".join(countries)
    text += f"Cluster {label} ({len(countries)} countries):\n"
    text += countries_str + "\n\n"

# Display colored legend patches on the side
ax.legend(handles=patches, title="Clusters", loc='upper right', fontsize=10)

# Display the country list text
ax.text(0, 1, text, fontsize=10, verticalalignment='top', wrap=True)

plt.show()


# In[101]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.patches as mpatches
import numpy as np

features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# First plot: PCA scatter with clustering
plt.figure(figsize=(12, 8))
cmap = plt.cm.viridis
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=70)
plt.title('K-Means Clustering Visualization with PCA (k=7)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)

# Metrics
ari = adjusted_rand_score(y_true.astype(int), y_pred)
nmi = normalized_mutual_info_score(y_true.astype(int), y_pred)
textstr = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"
plt.text(x=min(X_pca[:, 0]), y=max(X_pca[:, 1]) * 1.4, s=textstr, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7))
plt.ylim(top=max(X_pca[:, 1]) * 1.1)

plt.show()

# Second plot: Cluster-country mapping
fig, ax = plt.subplots(figsize=(6, 12))
ax.axis('off')

cluster_labels = range(7)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

# Print clusters and countries to console
for label in cluster_labels:
    countries = cluster_countries[label]
    print(f"Cluster {label} ({len(countries)} countries):")
    print(", ".join(countries))
    print("-" * 60)

# Display clusters with countries in matching colors
y_position = 1.0
for label in cluster_labels:
    countries = cluster_countries[label]
    color = cmap(label / (max(cluster_labels)))  # Normalize label for colormap
    country_text = f"Cluster {label} ({len(countries)}):\n" + ", ".join(countries)
    
    ax.text(0, y_position, country_text,
            fontsize=10, verticalalignment='top', color=color, wrap=True)
    
    # Adjust y position for next block of text
    y_position -= 0.15 + 0.015 * len(countries)

plt.tight_layout()
plt.show()



# In[85]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Features for clustering
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot clusters without cluster labels on points
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=70)
plt.title('K-Means Clustering Visualization with PCA (k=7)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)

# Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) text box (numeric values)
ari = 0.0133
nmi = 0.4924
textstr = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"
plt.text(x=min(X_pca[:,0]), y=max(X_pca[:,1]) * 1.4, s=textstr, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7))

plt.ylim(top=max(X_pca[:,1]) * 1.1)  # Increase top limit for text visibility

plt.show()


# Calculate and print clustering performance metrics
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")


# In[157]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

# Features
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]
y_true = df['GGGI Score'].astype('category').cat.codes  # True labels

# KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)

# Cluster label colors
cmap = plt.cm.viridis
cluster_sizes = np.bincount(y_pred)
cluster_labels = range(7)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

# --- Create side-by-side plots ---
fig, (ax_scatter, ax_text) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [3, 1]})

# --- Scatter plot (left), no labels ---
scatter = ax_scatter.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=70)

# Basic formatting (no text annotations)
ax_scatter.set_title('K-Means Clustering with PCA (k=7)', fontsize=14)
ax_scatter.set_xlabel('PCA Component 1')
ax_scatter.set_ylabel('PCA Component 2')
ax_scatter.grid(True)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax_scatter)
cbar.set_label('Cluster Label')

# --- Cluster label and country listing (right) ---
ax_text.axis('off')
y_pos = 1.0
line_height = 0.03

for label in cluster_labels:
    countries = cluster_countries[label]
    size = len(countries)
    color = cmap(label / (len(cluster_labels) - 1))
    
    # Başlık
    header = f"Cluster {label} ({size} countries):"
    ax_text.text(0, y_pos, header, fontsize=12, color=color, verticalalignment='top', weight='bold')
    y_pos -= line_height

    # Ülkeleri 4’erli satırlara böl
    for i in range(0, len(countries), 5):
        line = ", ".join(countries[i:i+5])
        ax_text.text(0.02, y_pos, line, fontsize=10, color='black', verticalalignment='top')
        y_pos -= line_height

    y_pos -= line_height * 0.5  # Clusterlar arası küçük boşluk

plt.tight_layout()
plt.show()



# ARI and NMI scores
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
metrics_text = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"
ax_scatter.text(min(X_pca[:, 0]), max(X_pca[:, 1]) * 1.3, metrics_text,
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# --- Cluster label and country listing (right) ---
ax_text.axis('off')
y_pos = 1.0
line_height = 0.03

for label in cluster_labels:
    countries = cluster_countries[label]
    size = len(countries)
    color = cmap(label / (len(cluster_labels) - 1))
      
    # Başlık
    header = f"Cluster {label} ({size} countries):"
    ax_text.text(0, y_pos, header, fontsize=12, color=color, verticalalignment='top', weight='bold')
    y_pos -= line_height

    # Ülkeleri 4’erli satırlara böl
    for i in range(0, len(countries), 4):
        line = ", ".join(countries[i:i+4])
        ax_text.text(0.02, y_pos, line, fontsize=10, color='black', verticalalignment='top')
        y_pos -= line_height

    y_pos -= line_height * 0.5  # Clusterlar arası küçük boşluk

# Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) text box
ari = 0.0133
nmi = 0.4924
textstr = f"Adjusted Rand Index (ARI): {ari:.4f}\nNormalized Mutual Information (NMI): {nmi:.4f}"

# 3D plot için metni bir kenara eklemek zor olabilir, bu yüzden figure üzerine ekleyelim
plt.figtext(0.15, 0.93, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.show()


# In[165]:


pip install squarify


# In[168]:


import squarify
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Patch

# Küme bilgileri
cluster_labels = range(7)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

# Treemap için veriler
labels = []
sizes = []
colors = []
cmap = cm.get_cmap('viridis', len(cluster_labels))

for label in cluster_labels:
    countries = cluster_countries[label]
    for country in countries:
        labels.append(f"{country}\n(C{label})")
        sizes.append(1)
        colors.append(cmap(label / (len(cluster_labels) - 1)))

# Treemap çizimi
fig, ax = plt.subplots(figsize=(16, 10))
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, text_kwargs={'fontsize': 8})
plt.axis('off')
plt.title("Cluster Treemap of Countries (KMeans, k=7)", fontsize=14)

# Lejand (küme renkleri)
legend_handles = [Patch(color=cmap(i / (len(cluster_labels) - 1)), label=f'Cluster {i}') for i in cluster_labels]
plt.legend(handles=legend_handles, loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# Konsola küme başına ülke listesi (n=... biçiminde)
print("\nCLUSTER ÜLKE LİSTESİ:\n" + "=" * 60)
for label in cluster_labels:
    countries = cluster_countries[label]
    print(f"Cluster {label} (n={len(countries)}):")
    print(", ".join(countries))
    print("-" * 60)


# In[181]:


import squarify
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Patch

# Cluster information
cluster_labels = range(7)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

# Data for treemap
labels = []
sizes = []
colors = []
cmap = cm.get_cmap('viridis', len(cluster_labels))

for label in cluster_labels:
    countries = cluster_countries[label]
    for country in countries:
        labels.append(f"{country}\n(C{label})")
        sizes.append(1)
        colors.append(cmap(label / (len(cluster_labels) - 1)))

# Plot treemap
fig, ax = plt.subplots(figsize=(16, 12))
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.8,
    text_kwargs={'fontsize': 11}  # Increased font size
)
ax.axis('off')
plt.title("Cluster Treemap of Countries (KMeans, k=7)", fontsize=14, pad=40)

# Legend (placed above the plot)
legend_handles = [
    Patch(
        color=cmap(i / (len(cluster_labels) - 1)),
        label=f'Cluster {i} (n={len(cluster_countries[i])})'
    ) for i in cluster_labels
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=4,
    fontsize=10,
    bbox_to_anchor=(0.3, 1.00),
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Print list of countries per cluster to console
print("\nCLUSTER COUNTRY LIST:\n" + "=" * 60)
for label in cluster_labels:
    countries = cluster_countries[label]
    print(f"Cluster {label} (n={len(countries)}):")
    print(", ".join(countries))
    print("-" * 60)


# In[ ]:





# In[163]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

# Özellikler
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]
y_true = df['GGGI Score'].astype('category').cat.codes

# KMeans
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# PCA (2B indirgeme)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(kmeans.cluster_centers_)

# Küme bilgileri
cmap = plt.cm.viridis
cluster_labels = range(7)
cluster_sizes = np.bincount(y_pred)
cluster_countries = {label: df.loc[y_pred == label, 'Country'].tolist() for label in cluster_labels}

# --- Yan yana görsel ve metin ---
fig, (ax_scatter, ax_text) = plt.subplots(1, 2, figsize=(14, 12), gridspec_kw={'width_ratios': [3, 1]})

# --- PCA scatter (sol) ---
scatter = ax_scatter.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=70)
ax_scatter.set_title('K-Means Clustering with PCA (k=7)', fontsize=14)
ax_scatter.set_xlabel('PCA Component 1')
ax_scatter.set_ylabel('PCA Component 2')
ax_scatter.grid(True)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax_scatter)
cbar.set_label('Cluster Label')

# --- Sağdaki metin kutusu ---
ax_text.axis('off')
y_pos = 1.0
line_height = 0.03

for label in cluster_labels:
    countries = cluster_countries[label]
    size = len(countries)
    color = cmap(label / (len(cluster_labels) - 1))

    # Başlık
    header = f"Cluster {label} ({size}):"
    ax_text.text(0, y_pos, header, fontsize=12, color=color, verticalalignment='top', weight='bold')
    y_pos -= line_height

    # Ülkeleri 4'erli gruplar hâlinde yaz
    for i in range(0, len(countries), 4):
        line = ", ".join(countries[i:i+4])
        ax_text.text(0.02, y_pos, line, fontsize=10, color='black', verticalalignment='top')
        y_pos -= line_height

    y_pos -= line_height * 0.5  # Küme arası boşluk

plt.tight_layout()
plt.show()

# Konsola da yaz
print("CLUSTER ÜLKE LİSTESİ:\n" + "="*60)
for label in cluster_labels:
    countries = cluster_countries[label]
    print(f"Cluster {label} ({len(countries)} countries):")
    print(", ".join(countries))
    print("-" * 60)


# In[ ]:





# In[ ]:





# In[111]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import pandas as pd

# Features
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# Sürekli hedef değişken
y_continuous = df['GGGI Score']

# Sürekli hedefi 3 sınıfa ayırıyoruz (kategorik hedef)
y_categorical = pd.qcut(y_continuous, q=3, labels=[0, 1, 2])

print("=== Regresyon Modelleri ===")
# Regresyon için veri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y_continuous, test_size=0.3, random_state=42)

regression_models = {
    "KNN Regressor": KNeighborsRegressor(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "Linear Regression": LinearRegression()
}

for name, model in regression_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE = {mse:.4f}, R2 = {r2:.4f}")

print("\n=== Sınıflandırma Modelleri ===")
# Sınıflandırma için veri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

classification_models = {
    "KNN Classifier": KNeighborsClassifier(),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "SVM Classifier": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

for name, model in classification_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")


# In[96]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Özellikler
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# KMeans clustering ile 7 küme bul
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X)

# Veri bölme (özellikler ve kümeler)
X_train, X_test, y_train, y_test = train_test_split(X, y_clusters, test_size=0.3, random_state=42)

# Modeller (ANN eklendi, Logistic Regression çıkarıldı)
classification_models = {
    "KNN Classifier": KNeighborsClassifier(),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "SVM Classifier": SVC(random_state=42),
    "ANN Classifier": MLPClassifier(max_iter=500, random_state=42)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(classification_models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performans metrikleri
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{name} Accuracy: {acc:.4f}, F1-Score: {f1:.4f}\n")

    # Confusion matrix çizimi, daha açık renk paleti
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r', ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{name}\nAccuracy: {acc:.4f}, F1-Score: {f1:.4f}", fontsize=12)
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.tight_layout()
plt.show()


# In[189]:


from sklearn.metrics import precision_score, recall_score

# Özellikler
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# KMeans clustering ile 7 küme bul
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X)

# Veri bölme (özellikler ve kümeler)
X_train, X_test, y_train, y_test = train_test_split(X, y_clusters, test_size=0.3, random_state=42)

# Modeller
classification_models = {
    "KNN Classifier": KNeighborsClassifier(),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "SVM Classifier": SVC(random_state=42),
    "ANN Classifier": MLPClassifier(max_iter=500, random_state=42)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(classification_models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performans metrikleri
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    print(f"\n{name} Performance Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Confusion matrix çizimi, mor tonları
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{name}\nAccuracy: {acc:.4f}, F1-Score: {f1:.4f}", fontsize=12)
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


print(df.columns)


# In[ ]:





# In[190]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Optimize edilecek modeller ve parametre ızgaraları
param_grid = {
    "KNN Classifier": {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance']
    },
    "Random Forest Classifier": {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10]
    },
    "SVM Classifier": {
        'model': [SVC(random_state=42)],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },
    "ANN Classifier": {
        'model': [MLPClassifier(max_iter=1000, random_state=42)],
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__solver': ['adam', 'lbfgs']
    }
}

from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer

# Performans skorları
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'F1': make_scorer(f1_score, average='weighted'),
    'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'Recall': make_scorer(recall_score, average='weighted')
}

best_models = {}

for name, params in param_grid.items():
    print(f"\nTuning {name}...")
    pipeline = Pipeline([('model', params['model'][0])])
    search = GridSearchCV(pipeline, {k: v for k, v in params.items() if k != 'model'}, 
                          scoring='accuracy', cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    # Skorlar
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')

    print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")


# In[191]:


import matplotlib.pyplot as plt
import seaborn as sns

# Grafik çizimi
fig, ax = plt.subplots(2, 2, figsize=(16, 12))

sns.barplot(data=results_df, x='Model', y='Accuracy', ax=ax[0, 0], palette='Set2')
ax[0, 0].set_title("Accuracy by Model")
for i, val in enumerate(results_df['Accuracy']):
    ax[0, 0].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='F1-Score', ax=ax[0, 1], palette='Set2')
ax[0, 1].set_title("F1-Score by Model")
for i, val in enumerate(results_df['F1-Score']):
    ax[0, 1].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='Precision', ax=ax[1, 0], palette='Set2')
ax[1, 0].set_title("Precision by Model")
for i, val in enumerate(results_df['Precision']):
    ax[1, 0].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='Recall', ax=ax[1, 1], palette='Set2')
ax[1, 1].set_title("Recall by Model")
for i, val in enumerate(results_df['Recall']):
    ax[1, 1].text(i, val + 0.01, f"{val:.2f}", ha='center')

plt.tight_layout()
plt.show()


# In[ ]:





# In[192]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# 1. Özellikleri belirle
# ---------------------------
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# ---------------------------
# 2. KMeans ile kümeleme
# ---------------------------
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X)

# ---------------------------
# 3. Eğitim-test veri bölme
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_clusters, test_size=0.3, random_state=42)

# ---------------------------
# 4. Parametre ızgaraları
# ---------------------------
param_grid = {
    "KNN Classifier": {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance']
    },
    "Random Forest Classifier": {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10]
    },
    "SVM Classifier": {
        'model': [SVC(random_state=42)],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },
    "ANN Classifier": {
        'model': [MLPClassifier(max_iter=1000, random_state=42)],
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__solver': ['adam', 'lbfgs']
    }
}

# ---------------------------
# 5. Skor metrikleri
# ---------------------------
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'F1': make_scorer(f1_score, average='weighted'),
    'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'Recall': make_scorer(recall_score, average='weighted')
}

best_models = {}
results = {
    'Model': [],
    'Accuracy': [],
    'F1-Score': [],
    'Precision': [],
    'Recall': []
}

# ---------------------------
# 6. GridSearchCV ile modellerin optimizasyonu
# ---------------------------
for name, params in param_grid.items():
    print(f"\nTuning {name}...")
    pipeline = Pipeline([('model', params['model'][0])])
    search = GridSearchCV(pipeline, {k: v for k, v in params.items() if k != 'model'},
                          scoring='accuracy', cv=5, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')

    print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    results['Model'].append(name)
    results['Accuracy'].append(acc)
    results['F1-Score'].append(f1)
    results['Precision'].append(prec)
    results['Recall'].append(rec)

# ---------------------------
# 7. DataFrame oluştur
# ---------------------------
results_df = pd.DataFrame(results)

# ---------------------------
# 8. Grafik ile sonuçların görselleştirilmesi
# ---------------------------
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Optimized Models Performance Comparison', fontsize=16)

sns.barplot(data=results_df, x='Model', y='Accuracy', ax=ax[0, 0], palette='Set2')
ax[0, 0].set_title("Accuracy by Model")
for i, val in enumerate(results_df['Accuracy']):
    ax[0, 0].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='F1-Score', ax=ax[0, 1], palette='Set2')
ax[0, 1].set_title("F1-Score by Model")
for i, val in enumerate(results_df['F1-Score']):
    ax[0, 1].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='Precision', ax=ax[1, 0], palette='Set2')
ax[1, 0].set_title("Precision by Model")
for i, val in enumerate(results_df['Precision']):
    ax[1, 0].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='Recall', ax=ax[1, 1], palette='Set2')
ax[1, 1].set_title("Recall by Model")
for i, val in enumerate(results_df['Recall']):
    ax[1, 1].text(i, val + 0.01, f"{val:.2f}", ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[ ]:





# In[194]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# 1. Define features
# ---------------------------
features = ['Economic Participation and Opportunity', 'Educational Attainment',
            'Health and Survival', 'Political Empowerment']
X = df[features]

# ---------------------------
# 2. Clustering with KMeans
# ---------------------------
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
y_clusters = kmeans.fit_predict(X)

# ---------------------------
# 3. Split data into train and test sets
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_clusters, test_size=0.3, random_state=42)

# ---------------------------
# 4. Parameter grids for models
# ---------------------------
param_grid = {
    "KNN Classifier": {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance']
    },
    "Random Forest Classifier": {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10]
    },
    "SVM Classifier": {
        'model': [SVC(random_state=42)],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },
    "ANN Classifier": {
        'model': [MLPClassifier(max_iter=1000, random_state=42)],
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__solver': ['adam', 'lbfgs']
    }
}

# ---------------------------
# 5. Scoring metrics
# ---------------------------
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'F1': make_scorer(f1_score, average='weighted'),
    'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'Recall': make_scorer(recall_score, average='weighted')
}

best_models = {}
results = {
    'Model': [],
    'Accuracy': [],
    'F1-Score': [],
    'Precision': [],
    'Recall': []
}

# ---------------------------
# 6. Optimize models with GridSearchCV
# ---------------------------
for name, params in param_grid.items():
    print(f"\nTuning {name}...")
    pipeline = Pipeline([('model', params['model'][0])])
    search = GridSearchCV(pipeline, {k: v for k, v in params.items() if k != 'model'},
                          scoring='accuracy', cv=5, n_jobs=-1)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')

    print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    results['Model'].append(name)
    results['Accuracy'].append(acc)
    results['F1-Score'].append(f1)
    results['Precision'].append(prec)
    results['Recall'].append(rec)

# ---------------------------
# 7. Create a results DataFrame
# ---------------------------
results_df = pd.DataFrame(results)

# ---------------------------
# 8. Visualize results with vibrant colors
# ---------------------------
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Optimized Models Performance Comparison', fontsize=16)

sns.barplot(data=results_df, x='Model', y='Accuracy', ax=ax[0, 0], palette='Set1')
ax[0, 0].set_title("Accuracy by Model")
for i, val in enumerate(results_df['Accuracy']):
    ax[0, 0].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='F1-Score', ax=ax[0, 1], palette='Set1')
ax[0, 1].set_title("F1-Score by Model")
for i, val in enumerate(results_df['F1-Score']):
    ax[0, 1].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='Precision', ax=ax[1, 0], palette='Set1')
ax[1, 0].set_title("Precision by Model")
for i, val in enumerate(results_df['Precision']):
    ax[1, 0].text(i, val + 0.01, f"{val:.2f}", ha='center')

sns.barplot(data=results_df, x='Model', y='Recall', ax=ax[1, 1], palette='Set1')
ax[1, 1].set_title("Recall by Model")
for i, val in enumerate(results_df['Recall']):
    ax[1, 1].text(i, val + 0.01, f"{val:.2f}", ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[ ]:




