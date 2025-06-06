import sys, os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *
import random

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

SEED = 42
random.seed(SEED)
print(f"SEED = {SEED}")


BONAFIDES = False # do not use the bonafide samples for attribution

Y = []
valid = []
c = 0
for fi in metadata:
    with open(os.path.join(meta_dir,fi)) as fin:
        for line in fin.readlines():
            if 'bonafide' not in line.strip().split('|')[2]:
                Y.append(fi.split('_')[0]+"_"+line.strip().split('|')[2])
                valid.append(c)
            c+=1

## Assign numeric labels to systems
systems = {k:i for i, k in enumerate(sorted(set(Y)))} 
print("Number of systems:", len(systems))
target_names = list(systems.keys())

## Assign labels to data
Y = np.array([systems[Y[i]] for i in range(len(Y))])

        
## Read the bert features
X = []
print("Reading features")
for fi in feats:
    x = np.load(os.path.join(feats_dir, fi))
    print(fi, x.shape)
    X.extend(x)

## Use only the fake data
X = np.array(X)
X = X[valid]


## Sanity check
print ("\nX shape:", X.shape)
print("Y shape: ", Y.shape)

## Train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=SEED)

print("Train size: ", Xtrain.shape[0])
print("Test size: ", Xtest.shape[0])

'''
print("\nScaling...")
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
'''


## Fit kNN
print("Fitting...")
K = 21
clf = KNeighborsClassifier(n_neighbors=K)#, weights='distance')
clf.fit(Xtrain, ytrain)

print("Predicting Xtest...")
Y_hat = clf.predict(Xtest)
print(classification_report(ytest, Y_hat, target_names=target_names))

with open("results_checkpoint_attribution.log", 'w') as fout:
    fout.write(f"Training data: {Xtrain.shape[0]} | Test data: {Xtest.shape[0]} \n")
    fout.write("------\n")

    fout.write(classification_report(ytest, Y_hat, target_names=target_names))
    fout.write("------\n")

    np.savetxt(fout, confusion_matrix(ytest, Y_hat, normalize='true'), delimiter=',', fmt='%.2f')

