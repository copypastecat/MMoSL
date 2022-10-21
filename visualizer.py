#code for visualizing

import numpy as np
import pandas as pd
from scipy import rand
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


data = pd.read_csv("project_train.csv")
features = data.iloc[:, :-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
labels = data.pop("Label")

embedded = TSNE(n_components=3,learning_rate='auto',init='pca',perplexity=5).fit_transform(features)
embedded_pca = PCA(n_components=3).fit_transform(features)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
emb_true = embedded[labels==1,:]
emb_false = embedded[labels==0,:]

ax.scatter(emb_true[:,0],emb_true[:,1],emb_true[:,2],marker='o')
ax.scatter(emb_false[:,0],emb_false[:,1],emb_false[:,2],marker='^')
plt.show()

lda = LinearDiscriminantAnalysis()
lda = lda.fit(features,labels)
#logreg = LogisticRegression(random_state=0)
#logreg.fit(features,labels)

lda_params = lda.get_params(deep=False)
print(lda_params)

#init classifier: stochastic gradient descent, batch size 1 (standard sgd). random_state for seeding initial weight initialization
#alpha=0 sets the regularization term to 0
clf = MLPClassifier(hidden_layer_sizes=(5,2),solver='sgd',batch_size=1,random_state=1,alpha=0)

