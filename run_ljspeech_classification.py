import sys, os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from config import *

dbs = {'asv19':0, 'asv21':1, 'timit':2,  'mlaad':3,  'asv5':4}
BONAFIDES = False
print(f"Using BONAFIDES={BONAFIDES}\n")

Y = []
valid = []
c = 0
for fi in metadata:
    with open(os.path.join(meta_dir,fi)) as fin:
        for line in sorted(fin.readlines()):
            if fi.split('_')[0] in dbs:
                    if 'bonafide' not in line.strip().split('|')[2]:
                         if ('ljspeech' in line) or ('LJSpeech' in line):
                             Y.append(line.split('|')[1])
                             valid.append(c)
            c+=1

Y = np.array(Y)
print("Y shape: ", Y.shape)

# Assign numeric labels to the different LJSpeech ckpts
labels = {k:i for i, k in enumerate(sorted(set(Y)))}
Y = np.array([labels[k] for k in Y])


# Read the feature files
X = []
for fi in feats:
    x = np.load(os.path.join(feats_dir, fi))
    print(fi, x.shape)
    X.extend(x)
X = np.array(X)

# Use only the fake and LJSpeech files
X = X[valid]
print ("\nX shape:", X.shape)


## Train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=0.1, random_state=42)

print("Train size: ", Xtrain.shape[0])
print("Test size: ", Xtest.shape[0])

'''
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
'''

print("Fitting...")
clf = KNeighborsClassifier(n_neighbors=21)
print(clf)
clf.fit(Xtrain, ytrain)


target_names = list(labels.keys())
print("Predicting Xtest...")
Y_hat = clf.predict(Xtest)
print(classification_report(ytest, Y_hat, target_names= list(labels.keys())))

'''
cm = confusion_matrix(ytest, Y_hat, normalize='true')
for k in range(cm.shape[0]):
    print(target_names[k], end=', ')
    for j in range(cm.shape[1]):
        print(np.round(cm[k][j],2), end=', ')
    print()
    
'''
