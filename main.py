# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from matplotlib import rcParams
rcParams['figure.figsize'] = 22, 10
RANDOM_SEED = 42
# %%
exercise_df = pd.read_csv('datasets/exercise.csv')
calories_df = pd.read_csv('datasets/calories.csv')
df = pd.concat([exercise_df, calories_df], axis=1)
df.head()
# %%
df.drop(['User_ID'], axis=1, inplace=True)
df.head()
# %%
df.isnull().sum()
# %%
sns.scatterplot(x='Age', y='Calories', data=df)
# %%
sns.scatterplot(x='Body_Temp', y='Calories', data=df)

# %%
sns.scatterplot(x='Weight', y='Calories', data=df)

# %%
sns.scatterplot(x='Height', y='Calories', data=df)
# %%
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df.head()
# %%
X = df.drop(['Calories'], axis=1)
Y = pd.DataFrame(df['Calories'])
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)
X.head()
Y.head()
# %%
X_scaler = RobustScaler()
Y_scaler = RobustScaler()
X = X_scaler.fit_transform(X)
Y = Y_scaler.fit_transform(Y)
# %%
X_pca = PCA()
Y_pca = PCA()
X = X_pca.fit_transform(X)
Y = Y_pca.fit_transform(Y)
# %%
print(Y[0])
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED)

# %%
reg = LinearRegression()
reg.fit(X_train, Y_train)
# %%
reg.score(X_train, Y_train)
# %%
pred = reg.predict(X_test)
# %%
inv_pred = Y_scaler.inverse_transform(pred)
inv_true = Y_scaler.inverse_transform(Y_test)
# %%


def scaled_mean_absolute_error(Y_true, Y_pred):
    return np.mean(np.abs((Y_pred-Y_true)))/np.mean(Y_true)


# %%
r2 = r2_score(Y_test, pred)
mae = scaled_mean_absolute_error(Y_test, pred)
print(r2, mae)
# %%
r2 = r2_score(inv_true, inv_pred)
mae = scaled_mean_absolute_error(inv_true, inv_pred)
print(r2, mae)
# %%
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, Y_train)
rf_reg.score(X_train, Y_train)
# %%
rf_pred = rf_reg.predict(X_test).reshape(-1, 1)
# %%
r2 = r2_score(Y_test, rf_pred)
print(r2)
# %%
inv_pca_true = Y_pca.inverse_transform(Y_test)
inv_true = Y_scaler.inverse_transform(inv_pca_true)
inv_pca_rf_pred = Y_pca.inverse_transform(rf_pred)
inv_rf_pred = Y_scaler.inverse_transform(inv_pca_rf_pred)
r2 = r2_score(inv_true, rf_pred)
print(r2)

# %%
plt.plot(inv_true.flatten(), marker='.', label='true')
plt.plot(inv_rf_pred.flatten(), 'r', marker='.', label='predicted')
plt.legend()
# %%
