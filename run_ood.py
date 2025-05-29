import os, random
import numpy as np
from config import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler



SEED = 42
random.seed(SEED)
print(f"SEED = {SEED}")



def compute_eer(Ytest, Y_hat):
    fpr, tpr, thresholds = roc_curve(Ytest, Y_hat, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print ("EER: ", np.round(eer*100,2), "| Threshold: ", np.round(thresh,3))
    return eer, thresh



class KNN_OOD_Detector:
    def __init__(self, k, threshold=None):
        self.k = k
        self.threshold = threshold
        self.neigh = None
        self.train_data = None
    
    def fit(self, X_train):
        """Fit the kNN model with training data."""
        self.train_data = X_train
        self.neigh = NearestNeighbors(n_neighbors=self.k)
        self.neigh.fit(X_train)
    
    def score_samples(self, X_test):
        """Compute the average distance of each test sample to its k nearest neighbors."""
        distances, _ = self.neigh.kneighbors(X_test)
        return np.mean(distances, axis=1)
    
    def predict(self, X_test):
        """Predict if a sample is in-distribution (0) or out-of-distribution (1)."""
        if self.threshold is None:
            raise ValueError("Threshold is not set. Please set a threshold before predicting.")
        
        scores = self.score_samples(X_test)
        return (scores > self.threshold).astype(int)
    
    def set_threshold(self, X_val, y_val):#, percentile=95):
        print("Set threshold over val samples", X_val.shape)
        """Set the threshold based on a validation set by computing the given percentile of scores."""
        scores = self.score_samples(X_val)
        eer, thresh = compute_eer(y_val, scores)
        self.threshold = thresh
       

    def evaluate(self, X_test, y_true):
        """Compute accuracy and F1 score for OOD detection."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, f1



def read_data(N=2):
    dbs = {'asv19':0, 'asv21':1, 'timit':2,  'mlaad':3,  'asv5':4}
    target_names =  ['asv19', 'asv21', 'timit', 'mlaad', 'asv5']

    db_systems =  {f:{} for f in dbs}
    systs = []

    Y = []
    valid = []
    bonafides = []
    db_labels = []
    c = 0
    for fi in metadata:
        with open(os.path.join(meta_dir,fi)) as fin:
            for line in fin.readlines():
                if fi.split('_')[0] in dbs:
                    Y.append(line.strip().split('|')[2])
                    db_labels.append(fi.split('_')[0])
                    if 'bonafide' not in line.strip().split('|')[2]:
                        valid.append(c)
                        db = fi.split('_')[0]
                        
                        syst = line.split('|')[2].strip()
                        if syst not in db_systems[db]:
                            db_systems[db][syst] = [c]
                        else:
                            db_systems[db][syst].append(c)
                        systs.append(syst)
                    else:
                        bonafides.append(c)
                    c+=1


    X = []
    for fi in feats:
        x = np.load(os.path.join(feats_dir, fi))
        print(fi, x.shape)
        X.extend(x)

    X = np.array(X)
    Y = np.array(Y)
    
    print ("\nX shape:", X.shape)    
    print ("\nY shape:", Y.shape)
    


    ## Set aside N systems to behave as OOD sources
    known_samples = []
    unk_train_samples, unk_test_samples, unk_val_samples = [], [], []
    unk_train_syst = []
    unk_test_syst = []
    unk_val_syst = []

    N = 4 # number of OOD systems per dataset (2 for val and 2 for test)
    for db in db_systems:
        all_ckpts = list(db_systems[db].keys())
        known_ckpts = random.sample(all_ckpts, len(all_ckpts)-N)
        for ckpt in known_ckpts:
            known_samples.extend(db_systems[db][ckpt])
        unk_ckpts = [x for x in all_ckpts if x not in known_ckpts]

        for k in range(len(unk_ckpts)//2):
            unk_val_samples.extend(db_systems[db][unk_ckpts[k]])
            unk_val_syst.append(unk_ckpts[k])

        for k in range(len(unk_ckpts)//2, len(unk_ckpts)):
            unk_test_samples.extend(db_systems[db][unk_ckpts[k]])
            unk_test_syst.append(unk_ckpts[k])


    train_samples = random.sample(known_samples, int(0.8*len(known_samples)))
    rem_samples = np.setdiff1d([x for x in range(X.shape[0])], bonafides)
    rem_samples = list(np.setdiff1d(rem_samples, train_samples))

    test_samples = random.sample(rem_samples, int(0.5*len(rem_samples)))
    val_samples = np.setdiff1d(rem_samples, test_samples)

    print("Overlaps:")
    print("Train-test", np.intersect1d(train_samples,test_samples))
    print("Train-val", np.intersect1d(train_samples, val_samples))
    print("Test-val", np.intersect1d(test_samples, val_samples))

    print("UNK train systs: ", len(unk_train_samples), unk_train_syst)
    print("UNK val systs: ", len(unk_val_samples), unk_val_syst)
    print("UNK test systs: ", len(unk_test_samples),  unk_test_syst)

    ## Assign numeric labels to system names                
    syst_keys = {}
    c = 1
    for db in db_systems:
        for ck in db_systems[db]:
            if ck not in unk_train_syst and ck not in unk_test_syst and ck not in unk_val_syst:
                syst_keys[ck] = 0 # in domain
                c+=1
            else:
                syst_keys[ck] = 1 #out of domain

    ## Get the subsets
    Xtrain = X[train_samples+unk_train_samples]
    ytrain = Y[train_samples+unk_train_samples]
    ytrain = np.array([syst_keys[k] for k in ytrain])

    Xtest = X[test_samples+unk_test_samples]
    ytest = Y[test_samples+unk_test_samples]
    ytest = np.array([syst_keys[k] for k in ytest])

    Xval = X[list(val_samples)+unk_val_samples]
    yval = Y[list(val_samples)+unk_val_samples]
    yval = np.array([syst_keys[k] for k in yval])

    print("TRAIN size and number of classes: ", Xtrain.shape[0], len(set(ytrain)))
    print("VAL size and number of classes: ", Xval.shape[0], len(set(yval)))
    print("TEST size and number of classes: ", Xtest.shape[0], len(set(ytest)))

    return Xtrain, ytrain, Xval, yval, Xtest, ytest, [db_labels[i] for i in test_samples+unk_test_samples]

            



if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, db_labels = read_data()

    # Fit the kNN OOD detector
    detector = KNN_OOD_Detector(k=1)
    print("Using K:=", detector.k)
    print(f"SEED = {SEED}")
    detector.fit(X_train)
    
    # Set threshold using a validation set
    print(set(y_val))
    detector.set_threshold(X_val,y_val)#, percentile=95)
    

    # Predict OOD samples
    print("Predicting over test")
    predictions = detector.predict(X_test)
    print("Predictions (0 = In domain, 1 = OOD):", predictions)

    print(classification_report(y_test, predictions, zero_division=1))

    for k in sorted(set(db_labels)):
        inds = [i for i in range(len(db_labels)) if db_labels[i]==k and y_test[i]==1]
        inds+= [i for i in range(y_test.shape[0]) if y_test[i]==0]
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(k, len(inds))
        print(classification_report(y_test[inds], predictions[inds], zero_division=1))

