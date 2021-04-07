from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import random
import numpy as np

#Ahora aplicamos la busqueda bayesiana para encontrar el mejor número de vecinos para el kNN.
def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return round(cross_val_score(clf, X, y, cv=5).mean(),4)
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,20))
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f, space4knn, algo=atpe.suggest, max_evals=100, trials=trials)
print('best:{}'.format(best))


f, ax = plt.subplots(1)#, figsize=(10,10))
xs = [t['misc']['vals']['n_neighbors'] for t in trials.trials]
ys = [-t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
ax.set_title('Iris Dataset - KNN', fontsize=18)
ax.set_xlabel('n_neighbors', fontsize=12)
ax.set_ylabel('cross validation accuracy', fontsize=12);










from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = SVC(**params)
    return round(cross_val_score(clf, X_, y).mean(),4)

space4svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=atpe.suggest, max_evals=100, trials=trials)
print('best:{}'.format(best))
parameters = ['C', 'kernel', 'gamma', 'scale', 'normalize']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
cmap = plt.cm.jet

for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=my_colors[i])
    axes[i].set_title(val)
    axes[i].set_ylim([0.9, 1.0])












from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X_original = iris.data
y_original = iris.target
def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = DecisionTreeClassifier(**params)
    return round(cross_val_score(clf, X, y).mean(),4)

space4dt = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}




def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4dt, algo=atpe.suggest, max_evals=100, trials=trials)
print('best:{}'.format(best))
















from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X_original = iris.data
y_original = iris.target

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = RandomForestClassifier(**params)
    return round(cross_val_score(clf, X, y).mean(),4)

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
}

best = 0
def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc > best:
        best = acc
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=100, trials=trials)
print(('best{}').format(best))











from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import accuracy_score

from tune_sklearn import TuneSearchCV

from mlxtend.plotting import plot_decision_regions

import warnings
'''
tune_search = TuneSearchCV(RandomForestClassifier(),
    param_distributions=param_dists,
    n_trials=10,
    scoring='accuracy',
    search_optimization="bayesian",
    random_state=1234,
    cv=5,
    n_jobs=-1,
    use_gpu=True
)'''

tune_search = TuneSearchCV(RandomForestClassifier(),
    param_distributions=param_dists,
    n_trials=10,
    scoring='accuracy',
    search_optimization="hyperopt",
    random_state=1234,
    cv=5,
    n_jobs=-1,
    use_gpu=True
)

tune_search.fit(X_train, y_train)
print(f'Best score: {tune_search.best_score_}', f'\nBest parameters: {tune_search.best_params_}') 




pred = tune_search.predict(X_test)
accuracy_score(y_test, pred)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)

best_params = tune_search.best_params_
best_params['max_features']=2

clf = RandomForestClassifier(**tune_search.best_params_)
clf.fit(principalComponents,y_train)

plot_decision_regions(X=principalComponents, y=y_train, clf=clf);