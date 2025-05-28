import sys, os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from config import *

dbs = ['mlaad', 'timit']
BONAFIDES = False

Y = []
valid = []
c = 0
for fi in metadata:
    with open(os.path.join(meta_dir,fi)) as fin:
        for line in sorted(fin.readlines()):
            db = fi.split('_')[0]
            if db in dbs:
                if 'bonafide' not in line.strip().split('|')[2]:
                    if db=='mlaad' and  'multilingual' in line:
                        Y.append(line.split('|')[1])
                        valid.append(c)
                    if db=='timit' and 'multi_speaker' in line:
                        Y.append(line.split('|')[1]+'_'+line.split('|')[2])
                        valid.append(c)
            c+=1

Y = np.array(Y)
print("Y shape: ", Y.shape)

labels = {k:i for i, k in enumerate(sorted(set(Y)))}
Y = np.array([labels[k] for k in Y])

# Read the features
X = []
for fi in feats:
    x = np.load(os.path.join(feats_dir, fi))
    print(fi, x.shape)
    X.extend(x)

X = np.array(X)
X = X[valid]
print ("\nX shape:", X.shape)


## Train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=22)

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

with open("results_multispeaker_systems_classification.txt", 'w') as fout:
    fout.write(f"Training data: {Xtrain.shape[0]} | Test data: {Xtest.shape[0]} \n")
    fout.write("------\n")

    fout.write(classification_report(ytest, Y_hat, target_names= list(labels.keys())))
    fout.write("------\n")

    np.savetxt(fout, confusion_matrix(ytest, Y_hat, normalize='true'), delimiter=',', fmt='%.2f')


