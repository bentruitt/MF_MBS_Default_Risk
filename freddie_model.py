from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from roc import plot_roc
from clean_data_2 import load_n_clean


#*** Load the dataset in with pandas
practice = False
if practice:
    trial = '_pract'
else:
    trial = '_final'

file_name = 'data/churn_train.csv'
df = load_n_clean(file_name)

#*** Make a numpy array called y containing the churn values
y = df.pop('churn').values

#*** Make a 2 dimensional numpy array containing the feature data (everything except #the labels)
X = df.values

#*** Use sklearn's train_test_split to split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)
orig_X_train, orig_X_test, orig_y_train, orig_y_test = X_train, X_test, y_train, y_test

if not practice:
    test_file = 'data/churn_test.csv'
    df_test = load_n_clean(test_file)
    y_test = df_test.pop('churn').values
    final_y_test = y_test
    orig_y_test = final_y_test
    X_test = df_test.values
    final_X_test = X_test
    orig_X_test = final_X_test

#*** Use sklearn's RandomForestClassifier to build a model of your data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# What is the accuracy score on the test data?
print "Random Forest Score:", rf.score(X_test, y_test), "\n"
## answer: 0.766856457896

#*** Try different splits
split_int = .10
for i in np.arange(split_int, 1.0, split_int):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, test_size=1-i)
    if not practice:
        y_test = final_y_test
        X_test = final_X_test
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print "Random Forest score with train size of: ", i, "score: ", rf.score(X_test, y_test)

# reset data to original splits
X_train, X_test, y_train, y_test = orig_X_train, orig_X_test, orig_y_train, orig_y_test
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#*** Draw a confusion matrix for the results
y_predict = rf.predict(X_test)
print "\nconfusion matrix:"
print confusion_matrix(y_test, y_predict)
## answer:  1950  743
##           921 3060

#*** What is the precision? Recall?
print "\nprecision:", precision_score(y_test, y_predict)
print "   recall:", recall_score(y_test, y_predict)
## precision: 0.798032107716
##    recall: 0.785222929936

#*** Build the RandomForestClassifier again setting the out of bag parameter to be true
rf = RandomForestClassifier(n_estimators=30, oob_score=True)
rf.fit(X_train, y_train)
print "\nwith out of bag:\n  accuracy score:", rf.score(X_test, y_test)
print "out of bag score:", rf.oob_score_
##   accuracy score: 0.766856457896
## out of bag score: 0.766856457896   (out-of-bag error is slightly worse)


#*** Use sklearn's model to get the feature importances
feature_importances = np.argsort(rf.feature_importances_)
print "\ntop five feature importances:", list(df.columns[feature_importances[-1:-6:-1]])
## top five: ['avg_dist', 'weekday_pct', 'signup_day', 'avg_rating_by_driver', ##'surge_pct']

# Calculate the standard deviation for feature importances across all trees

n = 15 # top 10 features

features = df.columns
importances = rf.feature_importances_[:n]
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(n):
    print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Importance of Each Feature")
plt.bar(range(n), importances[indices], yerr=std[indices], color="r", align="center")
plt.xticks(range(n), features, rotation=75)
plt.ylabel("Importance")
plt.xlim([-1, n])
plt.tight_layout()
plt.savefig("feature_importances" + trial + ".png")
plt.close()

#*** Try modifying the number of trees
num_trees = range(5, 100, 5)
accuracies = []
for n in num_trees:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
plt.figure()
plt.plot(num_trees, accuracies)
plt.title("Number of Trees vs. Accuracy")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("modifying_nm_trees" + trial + ".png")
plt.close()
# levels off around 77.7% at around 80 trees

#*** Try modifying the max features parameter
num_features = range(1, len(df.columns) + 1)
accuracies = []
for n in num_features:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(max_features=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
plt.figure()
plt.plot(num_features, accuracies)
plt.title("Number of Features vs. Accuracy")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("modifying_max_features" + trial + ".png")
plt.close()
## A lot of fluctuation. 3 features seems to be best.

'''
##*** Plot histograms of data for top 5 features
##
nm_feat = 5
plt.figure()
n, bins, patches = plt.hist(df[features[0]], 50, normed=1, alpha=0.75)
plt.tight_layout()
plt.savefig("case_histograms" + trial + ".png")
'''


#*** Run random forest in comparison to logistic regression, decision tree, SVM, and #Naive Bayes
def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

print "\n       Model            |  Accuracy     |    Precision     |      Recall"
print "    Random Forest:         %f           %f           %f" % get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
print "    Logistic Regression:   %f           %f           %f" % get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
print "    Decision Tree:         %f           %f           %f" % get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
print "    SVM:                   %f           %f           %f" % get_scores(SVC, X_train, X_test, y_train, y_test)
print "    Naive Bayes:           %f           %f           %f" % get_scores(MultinomialNB, X_train, X_test, y_train, y_test)
## MODEL               ACCURACY PRECISION    RECALL
## Random Forest:      0.7533     0.7814      0.8110
## Logistic Regression 0.7135     0.7254      0.8317
## Decision Tree       0.6841     0.7382      0.7241
## SVM:                0.7413     0.7693      0.8054
## Naive Bayes:        0.6514     0.6816      0.7736

#***Use plot_roc function provided during random forests to visualize curve of each #model
print "Use the `plot_roc` function to visualize the roc curve:"
plot_roc(X, y, trial, RandomForestClassifier, n_estimators=25, max_features=5)
plot_roc(X, y, trial, LogisticRegression)
plot_roc(X, y, trial, DecisionTreeClassifier)
#plot_roc(X, y, trial, SVC)
#plot_roc(X, y, trial, MultinomialNB)
plot_roc(X, y, trial, GradientBoostingClassifer, lr=0.025, n_estimators=500, random_state=1, min_samples_leaf=10, max_features=0.3, max_depth=6)
