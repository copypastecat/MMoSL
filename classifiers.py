import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("project_train.csv")
features = data.iloc[:, :-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
labels = data.pop("Label")
test_data = pd.read_csv("project_test.csv")
test_features = test_data.apply(lambda x: (x-x.mean())/x.std(), axis=0)

#init classifiers:
clf_nn = MLPClassifier(hidden_layer_sizes=(20,10,5),solver='sgd',batch_size=10,random_state=1,alpha=0.000,activation='identity')
clf_nn = clf_nn.fit(features,labels)
clf_svm = svm.SVC(kernel='rbf')
clf_svm = clf_svm.fit(features, labels)
clf_lr = LogisticRegression(solver='liblinear', C=1, random_state=0)
clf_lr.fit(features, labels)

#'''
#cross-validation
cv = RepeatedStratifiedKFold(n_splits=11, n_repeats=3, random_state=1)
scores_nn = cross_val_score(clf_nn, features, labels, scoring='accuracy', cv=cv, n_jobs=-1,verbose=False)
scores_svm = cross_val_score(clf_svm, features, labels, scoring='accuracy', cv=cv, n_jobs=-1,verbose=False)
scores_lr = cross_val_score(clf_lr, features, labels, scoring='accuracy', cv=cv, n_jobs=-1,verbose=False)
print("Mean accuracy score: NN: ", scores_nn.mean(), " ; SVM: ", scores_svm.mean(), " ; LogReg: ", scores_lr.mean())
print("Accuracy std-dev: NN: ", scores_nn.std(), " ; SVM: ", scores_svm.std(), " ; LogReg: ", scores_lr.std())
#'''
