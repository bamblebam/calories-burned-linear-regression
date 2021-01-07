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
print(Y[0])
# %%
