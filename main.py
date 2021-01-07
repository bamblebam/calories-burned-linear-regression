# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
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
