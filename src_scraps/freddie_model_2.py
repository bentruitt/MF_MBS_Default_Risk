import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import random
from roc import plot_roc
from scipy import interp
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

random.seed(1234)

#*** Run random forest in comparison to logistic regression, decision tree, SVM, and #Naive Bayes
def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

#*** Load data that was prepared by freddie_data_analysis module
def load_data(file):
    # read csv file prepared by freddie_data_analysis module
    df = pd.read_csv(file)
    # drop unneeded columns
    columns = df.columns.tolist()
    for col in columns:
        if 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df.drop(['published_date'], axis=1, inplace=True)
    # replace nan values with 0
    df.fillna(0, inplace=True)
    # apply get_dummies to particular columns
    df = pd.get_dummies(df, prefix=['state'], columns=['property_state'])
    # return prepared dataframe
    return df

def strat_kfold_split(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = StratifiedKFold(y, n_folds=2, shuffle=True)
    y_prob = np.zeros((len(y),2))
    # set_trace()
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test

def plot_roc_curve(X_train, X_test, y_train, y_test, plot_dir, trial, clf_class, **kwargs):

    model = clf_class(**kwargs)
    model.fit(X_train, y_train)
    y_predict_prob = model.predict_proba(X_test)[:,1]
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    model_nm = str(clf_class).split('.')[-1:][0].split("'")[0]
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot for ' + model_nm)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir + 'ROC_plot_' + model_nm + trial + '.png')
    plt.close()

if __name__ == '__main__':

    plot_dir = 'plots/'
    trial = '_version2_check'
    datadir = 'data/'

    #*** Load the dataset in with pandas and refine columns
    df = load_data(datadir + 'df_labeled' + '.csv')

    ### Seperate labels and feature data
    #*** Make a numpy array called y containing the default labels
    y = df.pop('label').values
    #*** Make a 2 dimensional numpy array containing the feature data (everything except #the labels)
    X = df.values

    #*** Use sklearn's train_test_split to split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y) #strat_kfold_split(X, y) #
    orig_X_train, orig_X_test, orig_y_train, orig_y_test = X_train, X_test, y_train, y_test

    #*** Select sklearn model to build
    model_name = LogisticRegression
    model = model_name(class_weight='balanced') #, , warm_start=True
    model_nm = str(model_name).split('.')[-1:][0].split("'")[0]
    model.fit(X_train, y_train)

    # What is the accuracy score on the test data?
    print model_nm + " Score:", model.score(X_test, y_test), "\n"
    ## Score: 0.988470249125

    #*** Draw a confusion matrix for the results
    use_prob=True
    y_predict_prob = model.predict_proba(X_test)[:,1]
    if use_prob:
        threshold = .5
        y_predict = (y_predict_prob > threshold).astype(float)
    else:
        y_predict = model.predict(X_test)
    print "\nconfusion matrix:\n"
    print pd.crosstab(pd.Series(y_test), pd.Series(y_predict), rownames=['True'], colnames=['Predicted'], margins=True)
    # print confusion_matrix(y_test, y_predict)
    # Predicted   0.0  1.0   All
    # True
    # 0.0        4795   29  4824
    # 1.0          27    6    33
    # All        4822   35  4857

    #*** What is the precision? Recall?
    print "\nprecision:", precision_score(y_test, y_predict)
    print "   recall:", recall_score(y_test, y_predict)
    # precision: 0.171428571429
    # recall: 0.181818181818

    # Rank top 'n' feature importances
    # n = 15
    #
    # features = df.columns
    # importances = model.feature_importances_[:n]
    # # std = np.std([tree.feature_importances_ for tree in model.estimators_],
    # #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    #
    # for f in range(n):
    #     print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

    # # Feature ranking:
    # # 1. current_balance (0.284215)
    # # 2. original_balance (0.236745)
    # # 3. o_int_rate (0.206056)
    # # 4. o_dsc_ratio (0.158775)
    # # 5. o_ltv_ratio (0.046985)
    # # 6. freddie_held (0.024425)
    # # 7. principal_paydown (0.003664)
    # # 8. state_AK (0.002428)
    # # 9. state_AL (0.000436)
    # # 10. state_AR (0.000193)
    # # 11. state_AZ (0.000000)
    # # 12. state_CA (0.000000)
    # # 13. state_CO (0.000000)
    # # 14. state_CT (0.000000)
    # # 15. state_DC (0.000000)
    #
    # # Plot the feature importances of the tree
    # plt.figure()
    # plt.title("Importance of Each Feature")
    # plt.bar(range(n), importances[indices], color="r", align="center") #yerr=std[indices],
    # plt.xticks(range(n), features, rotation=75)
    # plt.ylabel("Importance")
    # plt.xlim([-1, n])
    # plt.tight_layout()
    # plt.savefig(plot_dir + "feature_importances" + trial + ".png")
    # plt.close()
    #
    # ##*** Plot histograms of data for top 5 features
    # ##
    # nm_feat = 5
    # plt.figure()
    # n, bins, patches = plt.hist(df[features[0]], 50, normed=1, alpha=0.75)
    # plt.tight_layout()
    # plt.savefig(plot_dir + "case_histograms" + trial + ".png")

    # print "\n       Model            |  Accuracy     |    Precision     |      Recall"
    print "    Random Forest:         %f           %f           %f" % get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Logistic Regression:   %f           %f           %f" % get_scores(LogisticRegression, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Decision Tree:         %f           %f           %f" % get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Linear SVM:            %f           %f           %f" % get_scores(LinearSVC, X_train, X_test, y_train, y_test, class_weight='balanced', C=0.5)
    print "    Gradient Boosting:     %f           %f           %f" % get_scores(GradientBoostingClassifier, X_train, X_test, y_train, y_test)
    # print "    Naive Bayes:           %f           %f           %f" % get_scores(MultinomialNB, X_train, X_test, y_train, y_test)
    # ## MODEL               ACCURACY PRECISION    RECALL
    # # Random Forest:         0.993206           0.000000           0.000000
    # # Logistic Regression:   0.993206           0.000000           0.000000
    # # Decision Tree:         0.988264           0.166667           0.181818
    # # SVM:                   0.993206           0.000000           0.000000
    #
    #
    # # set_trace()
    # #***Use plot_roc function provided during random forests to visualize curve of each #model
    # print "Use the `plot_roc` function to visualize the roc curve:"
    # plot_roc_curve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, plot_dir=plot_dir, trial=trial, clf_class=RandomForestClassifier, class_weight='balanced')
    # plot_roc_curve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, plot_dir=plot_dir, trial=trial, clf_class=LogisticRegression, class_weight='balanced')
    # plot_roc_curve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, plot_dir=plot_dir, trial=trial, clf_class=DecisionTreeClassifier, class_weight='balanced')
    # # plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=LinearSVC, class_weight='balanced')
    # # plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=MultinomialNB)
    # plot_roc_curve(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, plot_dir=plot_dir, trial=trial, clf_class=GradientBoostingClassifier)
