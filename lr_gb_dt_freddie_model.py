import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import random
from roc import plot_roc
from scipy import interp
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
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

def plot_roc_curve(X, y, plot_dir, trial, cv, model):

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    model_nm = str(model).split("(")[0]

    for i, (train, test) in enumerate(cv):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot for ' + model_nm)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir + 'ROC_plot_' + model_nm + trial + '.png')
    plt.close()

def feature_importance():
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
    pass

def grid_search(X_train, X_test, y_train, y_test, cv, estimator, tuned_parameters):
    # Set the parameters by cross-validation
    scores = ['precision_macro', 'recall_macro', 'f1_weighted', 'f1']

    for score in scores:
        clf = GridSearchCV(estimator(), tuned_parameters, cv=cv, scoring=score)
        clf.fit(X_train, y_train)

        print "\n### Tuning hyper-parameters for %s" % score

        print "Best parameters set found on development set:"
        print clf.best_params_

        print "\nGrid scores on development set:"
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print "\nDetailed classification report:"
        print "\nThe model is trained on the full development set."
        print "The scores are computed on the full evaluation set.\n"

        y_true, y_pred = y_test, clf.predict(X_test)
        print classification_report(y_true, y_pred)
        print "\n-----------------------------------------------------------\n"

if __name__ == '__main__':

    plot_dir = 'plots/'
    trial = '_final1'
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

    ### Create StratifiedKFold generator
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X)
    cv = StratifiedKFold(y, n_folds=5, shuffle=True)

    #*** Build LogisticRegression model
    # LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    lr = LogisticRegression(penalty='l1', class_weight='balanced')
    lr.fit(X_train, y_train)

    tuned_parameters = [{
    'penalty': ['l1'],
    'C': [1., 10., 100., 1000.],
    'class_weight': ['balanced', None]
    }, {
    'penalty': ['l2'],
    'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear'],
    'C': [1., 10., 100., 1000.],
    'class_weight': ['balanced', None]
    }, {
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'dual': [True],
    'C': [1., 10., 100., 1000.],
    'class_weight': ['balanced', None]
    }]

    grid_search(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=5,  estimator=LogisticRegression, tuned_parameters=tuned_parameters)

    ## Determine optimal number of features
    # rfecv = RFECV(estimator=lr, step=1, cv=cv, scoring='f1')
    # rfecv.fit(X, y)
    # print("Optimal number of features for ", str(lr).split("(")[0], ": %d" % rfecv.n_features_)
    # # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()

    #*** Build DecisionTreeClassifier model
    # DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
    dt = DecisionTreeClassifier(class_weight='balanced') #, , warm_start=True
    dt.fit(X_train, y_train)

    tuned_parameters = [{
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': [5, 10, 20, 40, 50, None],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 3, 4],
    'min_weight_fraction_leaf': [0, .25, .50],
    'class_weight': [None, "balanced"]
    }]

    grid_search(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=5,  estimator=DecisionTreeClassifier, tuned_parameters=tuned_parameters)

    #*** Build GradientBoostingClassifier model
    # GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    gb = GradientBoostingClassifier() #, , warm_start=True
    gb.fit(X_train, y_train)

    tuned_parameters = [{
    'loss': ['deviance', 'exponential'],
    'n_estimators': [50, 100, 500],
    'max_depth': [3, 5, 10, 15, 20],
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'min_samples_split': [2, 3, 4],
    'min_weight_fraction_leaf': [0., .25, .50],
    'subsample': [.25, .50, .75, 1.],
    'max_features': [5, 10, 20, 40, 50, None],
    'init': [None, LogisticRegression]
    }]

    grid_search(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=5,  estimator=GradientBoostingClassifier, tuned_parameters=tuned_parameters)

    models = [lr, dt, gb]

    #*** Draw a confusion matrixes for the results
    use_prob=False
    threshold = .5
    y_prob = []
    y_pred = []
    for i, model in enumerate(models):
        model_nm = str(model).split("(")[0]
        y_prob.append(model.predict_proba(X_test)[:,1])
        if use_prob:
            threshold = .5
            y_pred.append((y_prob[i] > threshold).astype(float))
        else:
            y_pred.append(model.predict(X_test))
        print "\n", model_nm, ": confusion matrix:\n"
        print pd.crosstab(pd.Series(y_test), pd.Series(y_pred[i]), rownames=['True'], colnames=['Predicted'], margins=True)
        print "\n"

    print "\n       Model            |  Accuracy     |    Precision     |      Recall"
    print "    Logistic Regression:   %f           %f           %f" % (lr.score(X_test, y_test), precision_score(y_test, y_pred[0]), recall_score(y_test, y_pred[0]))
    print "    Decision Tree:         %f           %f           %f" % (dt.score(X_test, y_test), precision_score(y_test, y_pred[1]), recall_score(y_test, y_pred[1]))
    print "    Gradient Boosting:     %f           %f           %f" % (gb.score(X_test, y_test), precision_score(y_test, y_pred[2]), recall_score(y_test, y_pred[2]))
    # ## MODEL               ACCURACY PRECISION    RECALL
    # # Logistic Regression:   0.993206           0.000000           0.000000
    # # Decision Tree:         0.988264           0.166667           0.181818
    # # SVM:                   0.993206           0.000000           0.000000
    #
    #
    ### Use plot_roc function provided during random forests to visualize curve of each #model
    print "Use the `plot_roc_curve` function to visualize the roc curve:"
    ## LogisticRegression ROC plot
    plot_roc_curve(X=X_t, y=y, plot_dir=plot_dir, trial=trial, cv=cv, model=lr)
    ## DecisionTreeClassifier ROC plot
    plot_roc_curve(X=X_t, y=y, plot_dir=plot_dir, trial=trial, cv=cv, model=dt)
    ## GradientBoostingClassifier ROC plot
    plot_roc_curve(X=X_t, y=y, plot_dir=plot_dir, trial=trial, cv=cv, model=gb)
