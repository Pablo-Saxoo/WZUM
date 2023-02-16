import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pickle

data = pd.read_csv('2021.12.21_project_data.csv', sep='\t')
data.drop(data.columns[[0]], axis=1, inplace=True)

for i in range(52, 69):
    data.iloc[:, i] = data.iloc[:, i].astype(float)

data['API'] = data['API'].str.rstrip('%').astype(float) / 100
data['SBI'] = data['SBI'].str.rstrip('%').astype(float) / 100

data.drop(['plec', 'wiek'], axis=1, inplace=True)

meanval = []
vals = [11, 16, 24, 31, 36, 44]
letters = ["B", "P"]

for i in range(len(data['GI - 16'])):
    data.at[i, 'API'] = 1 - data.at[i, 'API']
    data.at[i, 'SBI'] = 1 - data.at[i, 'SBI']

for val in vals:
    for letter in letters:
        data[f'Interleukina – {val}{letter}'] = data[f'Interleukina – {val}{letter}'].fillna(
            data[f'Interleukina – {val}{letter}'].mean())

df_target = data[data.columns[:12]]
fn = 'processing/'
for val in vals:
    for letter in letters:
        df_target_val = df_target[f"{val}-{letter}"]
        X_train, X_test, y_train, y_test = train_test_split(data.drop(data.iloc[:, 0:12], axis=1), df_target_val,
                                                            test_size=0.2, random_state=42)

        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # with open(f'scal{val}-{letter}', 'wb') as file:
        #     pickle.dump(scaler, file)

        with open(f'{fn}scal{val}-{letter}', 'rb') as file:
            scl = pickle.load(file)

        X_train = scl.transform(X_train)
        X_test = scl.transform(X_test)

        clf = svm.LinearSVR(C=10)
        clf.fit(X_train, y_train)

        # with open(f'clf{val}-{letter}', 'wb') as file:
        #     pickle.dump(clf, file)

        val_min = data[f'{val}-{letter}'].min()
        val_max = data[f'{val}-{letter}'].max()
        val_diff = val_max - val_min
        meanval.append(mean_absolute_error(y_test, clf.predict(X_test)))
        print(
            f"For {val}-{letter}: {mean_absolute_error(y_test, clf.predict(X_test)):<18} (min: {val_min:<2}, max: {val_max:<3}, diff: {val_diff:<3})")

print(np.mean(meanval))