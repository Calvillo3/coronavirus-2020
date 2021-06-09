"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. Take the difference between cases. Get
the distribution of this data (to make it time-invariant).
Use the distribution as the feature vector.
"""

import sys
sys.path.insert(0, '..')

from utils import data
import matplotlib.pyplot as plt
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 20
MIN_CASES = 5000
N_BINS = 20
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

features = np.concatenate(features, axis=0)
targets = np.concatenate(targets, axis=0)
predictions = {}

for _dist in ['minkowski', 'manhattan']:
    for val in np.unique(confirmed["Country/Region"]):
        # test data
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        # filter the rest of the data to get rid of the country we are
        # trying to predict
        mask = targets[:, 1] != val
        tr_features = features[mask]
        tr_targets = targets[mask][:, 1]

        above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
        tr_features = np.diff(tr_features[above_min_cases], axis=-1)
        
        if NORMALIZE:
            tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)
        tr_features = np.apply_along_axis(
            lambda a: np.histogram(a, bins=N_BINS)[0], -1, tr_features)
        
        tr_targets = tr_targets[above_min_cases]

        # train knn
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
        knn.fit(tr_features, tr_targets)

        # predict
        cases = cases.sum(axis=0, keepdims=True)
        cases = np.apply_along_axis(
            lambda a: np.histogram(a, bins=N_BINS)[0], -1, cases)
        # nearest country to this one based on trajectory
        label = knn.predict(cases)
        
        if val not in predictions:
            predictions[val] = {}
        predictions[val][_dist] = label.tolist()
predictioncount = {}
for i in predictions:
    if predictions[i]["minkowski"][0] not in predictioncount:
        predictioncount[predictions[i]["minkowski"][0]] = 1
    else:
        predictioncount[predictions[i]["minkowski"][0]] += 1
print(predictioncount)
plt.bar(range(len(predictioncount)), list(predictioncount.values()), align='center')
plt.xticks(range(len(predictioncount)), list(predictioncount.keys()), rotation=55)
plt.ylabel("Number of matches")
plt.title("Matches/Country-dist_diff (N neighbors = " + str(N_NEIGHBORS) + ", min cases =" + str(MIN_CASES) +")")
#plt.legend([N_NEIGHBORS, MIN_CASES], ["N neighbors", "min cases"])
plt.show()

with open('results/knn_dist_diff.json', 'w') as f:
    json.dump(predictions, f, indent=4)
