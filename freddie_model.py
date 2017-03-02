from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from roc import plot_roc
from pdb import set_trace
import random

random.seed(1234)

#*** Run random forest in comparison to logistic regression, decision tree, SVM, and #Naive Bayes
def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

if __name__ == '__main__':

    plot_dir = 'plots/'
    trial = '_error_checking'
    #*** Load the dataset in with pandas
    datadir = 'data/'
    df = pd.read_csv(datadir + 'df_labeled' + '.csv')
    columns = df.columns.tolist()
    for col in columns:
        if 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df.drop(['published_date'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, prefix=['state'], columns=['property_state'])
    #*** Make a numpy array called y containing the default labels
    y = df.pop('label').values

    #*** Make a 2 dimensional numpy array containing the feature data (everything except #the labels)
    X = df.values
    # set_trace()
    #*** Use sklearn's train_test_split to split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    orig_X_train, orig_X_test, orig_y_train, orig_y_test = X_train, X_test, y_train, y_test

    #*** Select sklearn model to build
    model_name = GradientBoostingClassifier
    model = model_name() #class_weight='balanced' 
    model_nm = str(model_name).split('.')[-1:][0].split("'")[0]
    model.fit(X_train, y_train)

    # What is the accuracy score on the test data?
    print model_nm + " Score:", model.score(X_test, y_test), "\n"
    ## Score: 0.988470249125

    # *** Try different splits
    # split_int = .25
    # for i in np.arange(split_int, 1.0, split_int):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, test_size=1-i, stratify=y)
    #     model.fit(X_train, y_train)
    #     print model_name + " score with train size of: ", i, "score: ", model.score(X_test, y_test)
    #
    # # reset data to original splits
    # X_train, X_test, y_train, y_test = orig_X_train, orig_X_test, orig_y_train, orig_y_test
    # model.fit(X_train, y_train)

    #*** Draw a confusion matrix for the results
    use_prob=False
    if use_prob:
        y_predict_prob = model.predict_proba(X_test)[:,1]
        threshold = .7
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



    #*** Build the RandomForestClassifier with the out of bag parameter to be true
    # rf = RandomForestClassifier(n_estimators=30, oob_score=True, class_weight='balanced')
    # rf.fit(X_train, y_train)
    # print "\nRandomForestClassifier with out of bag:\n  accuracy score:", rf.score(X_test, y_test)
    # print "out of bag score:", rf.oob_score_
    # accuracy score: 0.992999794112
    # out of bag score: 0.993066996156



    #*** Use sklearn's model to get the feature importances
    # feature_importances = np.argsort(model.feature_importances_)
    # print "\ntop five feature importances:", list(df.columns[feature_importances[-1:-6:-1]]), "\n"
    # ## top five: ['o_int_rate', 'original_balance', 'o_dsc_ratio', 'o_ltv_ratio', 'principal_paydown']

    # Calculate the standard deviation for feature importances across all trees

    # n = 15 # top 10 features
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

    # Feature ranking:
    # 1. current_balance (0.284215)
    # 2. original_balance (0.236745)
    # 3. o_int_rate (0.206056)
    # 4. o_dsc_ratio (0.158775)
    # 5. o_ltv_ratio (0.046985)
    # 6. freddie_held (0.024425)
    # 7. principal_paydown (0.003664)
    # 8. state_AK (0.002428)
    # 9. state_AL (0.000436)
    # 10. state_AR (0.000193)
    # 11. state_AZ (0.000000)
    # 12. state_CA (0.000000)
    # 13. state_CO (0.000000)
    # 14. state_CT (0.000000)
    # 15. state_DC (0.000000)

    # Plot the feature importances of the tree
    # plt.figure()
    # plt.title("Importance of Each Feature")
    # plt.bar(range(n), importances[indices], color="r", align="center") #yerr=std[indices],
    # plt.xticks(range(n), features, rotation=75)
    # plt.ylabel("Importance")
    # plt.xlim([-1, n])
    # plt.tight_layout()
    # plt.savefig(plot_dir + "feature_importances" + trial + ".png")
    # plt.close()

    #*** Try modifying the number of trees in a random forest
    # num_trees = range(5, 60, 5)
    # accuracies = []
    # for n in num_trees:
    #     tot = 0
    #     for i in xrange(5):
    #         rf = RandomForestClassifier(n_estimators=n, class_weight='balanced')
    #         rf.fit(X_train, y_train)
    #         tot += rf.score(X_test, y_test)
    #     accuracies.append(tot / 5)
    # plt.figure()
    # plt.plot(num_trees, accuracies)
    # plt.title("Number of Trees in Forest vs. Accuracy")
    # plt.xlabel("Number of Trees")
    # plt.ylabel("Accuracy")
    # plt.tight_layout()
    # plt.savefig(plot_dir + "modifying_nm_trees" + trial + ".png")
    # plt.close()
    # # levels off around 77.7% at around 80 trees

    #*** Try modifying the max features parameter
    # num_features = range(1, len(df.columns) + 1)
    # accuracies = []
    # for n in num_features:
    #     tot = 0
    #     for i in xrange(5):
    #         model = model_name(max_features=n)
    #         model.fit(X_train, y_train)
    #         tot += model.score(X_test, y_test)
    #     accuracies.append(tot / 5)
    # max_features = np.argmax(np.array(accuracies))+1
    # plt.figure()
    # plt.plot(num_features, accuracies)
    # plt.title("Number of Features in Tree vs. Accuracy")
    # plt.xlabel("Number of Features")
    # plt.ylabel("Accuracy")
    # plt.tight_layout()
    # plt.savefig(plot_dir + "modifying_max_features" + trial + ".png")
    # plt.close()
    ## A lot of fluctuation. 3 features seems to be best.

    ##*** Plot histograms of data for top 5 features
    ##
    # nm_feat = 5
    # plt.figure()
    # n, bins, patches = plt.hist(df[features[0]], 50, normed=1, alpha=0.75)
    # plt.tight_layout()
    # plt.savefig(plot_dir + "case_histograms" + trial + ".png")

    print "\n       Model            |  Accuracy     |    Precision     |      Recall"
    print "    Random Forest:         %f           %f           %f" % get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Logistic Regression:   %f           %f           %f" % get_scores(LogisticRegression, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Decision Tree:         %f           %f           %f" % get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Linear SVM:            %f           %f           %f" % get_scores(LinearSVC, X_train, X_test, y_train, y_test, class_weight='balanced')
    print "    Gradient Boosting:     %f           %f           %f" % get_scores(GradientBoostingClassifier, X_train, X_test, y_train, y_test)
    # print "    Naive Bayes:           %f           %f           %f" % get_scores(MultinomialNB, X_train, X_test, y_train, y_test)
    ## MODEL               ACCURACY PRECISION    RECALL
    # Random Forest:         0.993206           0.000000           0.000000
    # Logistic Regression:   0.993206           0.000000           0.000000
    # Decision Tree:         0.988264           0.166667           0.181818
    # SVM:                   0.993206           0.000000           0.000000


    # set_trace()
    #***Use plot_roc function provided during random forests to visualize curve of each #model
    print "Use the `plot_roc` function to visualize the roc curve:"
    plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=RandomForestClassifier, class_weight='balanced')
    plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=LogisticRegression, class_weight='balanced')
    plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=DecisionTreeClassifier, class_weight='balanced')
    # plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=LinearSVC, class_weight='balanced')
    # plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=MultinomialNB)
    plot_roc(X=X, y=y, plot_dir=plot_dir, trial=trial, clf_class=GradientBoostingClassifier)
