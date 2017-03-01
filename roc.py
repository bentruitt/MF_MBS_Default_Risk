import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pdb import set_trace
import random

random.seed(1234)


def plot_roc(X, y, plot_dir, trial, clf_class, **kwargs):
    # set_trace()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = StratifiedKFold(y, n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        #pdb.set_trace()
        try:
            y_prob[test_index] = clf.predict_proba(X_test)
        except:
            print "No true-positives calculated / No probability for ", clf_class
            return
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    model_nm = str(clf_class).split('.')[-1:][0].split("'")[0]
    plt.title('ROC plot for ' + model_nm)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir + 'ROC_plot_' + model_nm + trial + '.png')
    plt.close()
