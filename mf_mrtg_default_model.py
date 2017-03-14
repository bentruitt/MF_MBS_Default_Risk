import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import random
from roc import plot_roc
from scipy import interp
from collections import Counter
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
from datetime import datetime

random.seed(1234)

### Run random forest in comparison to logistic regression, decision tree, SVM, and #Naive Bayes
def get_scores(model, X_test, y_train, y_test, y_pred):
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = (precision + recall) / 2.
    return accuracy, \
           precision, \
           recall, \
           f1_score

### Load data that was prepared by freddie_data_analysis module
def load_data(in_file):
    # read csv file prepared by freddie_data_analysis module
    df = pd.read_csv(in_file)
    # drop unneeded columns
    columns = df.columns.tolist()
    for col in columns:
        if 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df.drop(['published_date'], axis=1, inplace=True)
    # replace nan values with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # apply get_dummies to particular columns
    df = pd.get_dummies(df, prefix=['state'], columns=['property_state'])
    df = pd.get_dummies(df, prefix=['ss'], columns=['special_servicer'])
    # return prepared dataframe
    return df

def plot_roc_curve(X, y, plot_dir, trial, cv, model):

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    thresh_plt = 0.0
    thresh_mean = 0.0
    model_nm = str(model).split("(")[0]
    ### Create StratifiedKFold generator
    cv = StratifiedKFold(y, n_folds=5, shuffle=True)
    ### Initialize StandardScaler
    scaler = StandardScaler()

    for i, (train, test) in enumerate(cv):
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        probas_ = model.fit(X_train, y[train]).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
        thresholds[0] = min(1.0, thresholds[0])
        thresholds[-1] = max(0.0, thresholds[-1])
        thresh_mean += interp(mean_fpr, np.linspace(0,1,len(thresholds)), thresholds)
        # plt.plot(fpr, thresholds, lw=1, label='Thresholds %d (%0.2f - %0.2f)' % (i+1, thresholds.max(), thresholds.min())) # np.linspace(0,1,len(thresholds))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')

    thresh_mean /= len(cv)
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot(mean_fpr, thresh_mean, 'k--', label='Mean Threshold')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot for ' + model_nm)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir + 'ROC_plot_' + model_nm + trial + '.png')
    plt.close()

def plot_feature_importance():

    pass

def grid_search(X_train, X_test, y_train, y_test, cv, estimator, scores, tuned_parameters, print_file):
    # Set the parameters by cross-validation
    model_nm = str(estimator).split('.')[-1:][0].split("'")[0]
    best_parameters = {}

    print_file.write("\n####--------------------%s Grid Search----------#####\n" % model_nm)
    for score in scores:
        clf = GridSearchCV(estimator(), tuned_parameters, n_jobs=8, cv=cv, scoring=score)
        clf.fit(X_train, y_train)

        print_file.write("\n### Tuning hyper-parameters for %s" % score)

        print_file.write("\nBest parameters set found on development set for %s: " % model_nm)
        print_file.write("%s" % clf.best_params_)
        best_parameters[score] = clf.best_params_

        print_file.write("\nGrid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print_file.write("\n%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print_file.write("\nDetailed classification report: for %s, %s" % (model_nm, score))
        print_file.write("\nThe model is trained on the full development set.")
        print_file.write("\nThe scores are computed on the full evaluation set.\n")

        y_true, y_pred = y_test, clf.predict(X_test)
        print_file.write(classification_report(y_true, y_pred))
        print_file.write("\n-----------------------------------------------------------\n")
    return best_parameters

def bootstrap_it(X, y, data_set_size=10000, true_proportion=.25):
    ### This function will take an unbalanced dataset and will split the data set into a true and false set. It will then randomly choose samples (with replacement) in even proportions until a desired size dataset is reached
    # X is dataset
    # y is labels
    # create true and false indexes to separate data
    if data_set_size < 5000:
        print "data_set_size is too small. Increasing to 5,000 samples."
        data_set_size=5000
    t_idx = y==1.0
    f_idx = y==0.0
    # separate data into true and false sets
    X_t, y_t = X[t_idx], y[t_idx]
    X_f, y_f = X[f_idx], y[f_idx]
    # randomly sample half of the desired set from each; the true and false sets
    t_sample = np.random.choice(len(X_t),int(true_proportion*data_set_size))
    X_t_s, y_t_s = X_t[t_sample], y_t[t_sample]
    f_sample = np.random.choice(len(X_f),int((1.-true_proportion)*data_set_size))
    X_f_s, y_f_s = X_f[f_sample], y_f[f_sample]
    # combine true and false portions into new dataset
    X_bs = np.concatenate((X_t, X_f), axis=0)
    y_bs = np.concatenate((y_t, y_f), axis=0)
    # shuffle arrays
    rng_state = np.random.get_state()
    np.random.shuffle(X_bs)
    np.random.set_state(rng_state)
    np.random.shuffle(y_bs)

    return X_bs, y_bs


if __name__ == '__main__':

    plot_dir = 'plots/'
    trial = '_f7'
    datadir = 'data/'
    file_options = ['df_comb_labeled', 'df_mspd_labeled_built_up', 'df_mflp_labeled', 'df_mspd_labeled']
    model_options = ['LogisticRegression', 'DecisionTreeClassifier', 'GradientBoostingClassifier']
    drop_freddie = False
    start_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M'))

    ### Select dataset to run
    for i, option in enumerate(file_options):
        selected = str(raw_input("Would you like to open %s?[y/n]" % option))
        if selected == 'y':
            if 'comb' not in option: drop_freddie = True
            file_to_load = option + '.csv'
            break
    if selected != 'y':
        print("Pickers can't be choosie! You get %s:)" % file_options[0])
        file_to_load = file_options[0] + '.csv'

    ### set trial name for file names
    trial += "_" + file_to_load.split("_")[1]
    if len(file_to_load.split("_"))>3: trial += "_" + "_".join(file_to_load.split(".")[0].split("_")[3:5])

    ### Open file for printing output
    print_file = open("output/" + trial + start_time + ".txt", "w")
    print_file.write("This analysis was run on the dataset from %s.\n" % (datadir + file_to_load))

    ### Load the dataset in with pandas and refine columns
    df = load_data(datadir + file_to_load)
    if drop_freddie:
        df.drop(['freddie_held'], axis=1, inplace=True)
    print_file.write("\nShape of DataFrame: (%d, %d)\n" % df.shape)
    print_file.write("Features in Dataset:\n")
    print_file.write(str(df.columns.tolist()))
    print_file.write("\n")

    ### Seperate labels and feature data
    ## Make a numpy array called y containing the default labels
    y = df.pop('label').values
    ## Make a numpy array containing 'loan_id' columns
    loan_ids = df.pop('loan_id').values
    ## Make a 2 dimensional numpy array containing the feature data (everything except #the labels)
    X = df.values

    ### Randomly sample data to adjust proportion of true values by oversampling
    boot_y = bool(raw_input("Would you like to randomly sample by class weighting?")=='y')
    trial += "_" + "boot_"
    if boot_y:
        data_set_size = int(raw_input("How large of a dataset would you like to create by random sampling?"))
        true_proportion = float(raw_input("What proportion would you like to be true values?[0.xxx]"))
        boot_y_bef = bool(raw_input("Would you like to randomly sample by class weighting before train_test_split?")=='y')
        if boot_y_bef:
            trial += "_" + "bef_"
            ## Boostrap random samples
            X_bs, y_bs = bootstrap_it(X, y, data_set_size, true_proportion)
            X, y = X_bs, y_bs
            ##  Use sklearn's train_test_split to split into train and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.33)
            orig_X_train, orig_X_test, orig_y_train, orig_y_test = X_train, X_test, y_train, y_test
        else:
            trial += "_" + "aft_"
            ## Use sklearn's train_test_split to split into train and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.33)
            orig_X_train, orig_X_test, orig_y_train, orig_y_test = X_train, X_test, y_train, y_test
            ## Boostrap random samples
            X_bs, y_bs = bootstrap_it(X_train, y_train, data_set_size, true_proportion)
            X_train, y_train = X_bs, y_bs
        print_file.write("\n\nThis is a randomly sampled training dataset with %d rows.\n" % data_set_size)
        print_file.write("\nThis randomly sampled dataset is sampled with a ratio of %f true labeled samples.\n" % true_proportion)
    else:
        ## Use sklearn's train_test_split to split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.33)
        orig_X_train, orig_X_test, orig_y_train, orig_y_test = X_train, X_test, y_train, y_test

    ### Apply StandardScaler
    apply_scaler = str(raw_input("Would you like to apply StandardScaler to X?[y/n]"))
    trial += "_" + apply_scaler
    if apply_scaler == 'y':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_t = scaler.fit_transform(X)
    else:
        X_t = X

    ### Check which models and grid searches to run
    scores = ['f1'] # 'precision_macro', 'recall_macro', 'f1_weighted'
    run_y = []
    grid_y = []
    models = []
    model_nms = []
    best_parameters_lr = {}
    best_parameters_dt = {}
    best_parameters_gb = {}

    for i, model in enumerate(model_options):
        run_y.append(bool(raw_input("Run %s?[y/n]" %(model))=='y'))
        if run_y[i]:
            grid_y.append((raw_input("Run %s GridSearchCV?[y/n]" %(model))=='y'))

    ### Build LogisticRegression model
    # LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    if run_y[0]:
        if grid_y[0]:
            tuned_parameters_lr = [{
            'penalty': ['l1'],
            'C': [1., 2500., 5000., 7500., 10000.],
            'class_weight': ['balanced', None]
            }, {
            'penalty': ['l2'],
            'solver': ['newton-cg'],
            'C': [1., 10., 100., 1000., 2000., 5000., 7500., 10000.],
            'class_weight': ['balanced', None]
            }]

            best_parameters_lr = grid_search(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=5,  estimator=LogisticRegression, tuned_parameters=tuned_parameters, scores = scores, print_file=print_file)
            print("Best parameters: %s" % str(best_parameters_lr))
            print_file.write("Best parameters: %s" % str(best_parameters_lr))
        else:
            best_parameters_lr['f1'] = {
            'penalty': 'l2',
            'C': 15000.0,
            'solver': 'newton-cg',
            'class_weight': 'balanced'
            }

        lr = LogisticRegression(**best_parameters_lr['f1'])
        lr.fit(X_train, y_train)
        models.append(lr)

    ### Build DecisionTreeClassifier model
    # DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
    if run_y[1]:
        if grid_y[1]:
            tuned_parameters = [{
            'criterion': ['gini'],
            'splitter': ['best', 'random'],
            'max_features': [10, 20, 40, 50, 55, None],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [3, 4, 5],
            'min_weight_fraction_leaf': [0],
            'class_weight': [None, "balanced"]
            }]

            best_parameters_dt = grid_search(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=5,  estimator=DecisionTreeClassifier, tuned_parameters=tuned_parameters, scores = scores, print_file=print_file)
            print("Best parameters: %s" % str(best_parameters_dt))
            print_file.write("Best parameters: %s" % str(best_parameters_dt))
        else:
            best_parameters_dt['f1'] = {
            'splitter': 'best',
            'min_samples_split': 4,
            'min_weight_fraction_leaf': 0,
            'criterion': 'gini',
            'max_features': None,
            'max_depth': None,
            'class_weight': None
            }

        dt = DecisionTreeClassifier(**best_parameters_dt['f1']) #, , warm_start=True
        dt.fit(X_train, y_train)
        models.append(dt)

    ### Build GradientBoostingClassifier model
    # GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    if run_y[2]:
        if grid_y[2]:
            tuned_parameters = [{
            'loss': ['deviance'], # ,'exponential'
            'n_estimators': [100], # 50, 200
            'max_depth': [15], #3, 5, 10, 15, 20
            'criterion': ['friedman_mse'], # ,'mse', 'mae'
            'min_samples_split': [3], # 3, 4
            'min_samples_leaf': [1], #, 2, 3
            'subsample': [1.0], # , .8, .9, 1.
            'max_features': [None], # 5, 10, 20, 40, 50, 55, None
            'init': [None], # 'deviance', 'exponential'
            'warm_start': [False, True]
            }]

            best_parameters_gb = grid_search(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=5,  estimator=GradientBoostingClassifier, tuned_parameters=tuned_parameters, scores = scores, print_file=print_file)
            print("Best parameters: %s" % str(best_parameters_gb))
            print_file.write("Best parameters: %s" % str(best_parameters_gb))
        else:
            best_parameters_gb['f1'] = {
            'loss': 'deviance',
            'min_samples_leaf': 1,
            'n_estimators': 100,
            'subsample': 0.7,
            'init': None,
            'min_samples_split': 3,
            'criterion': 'friedman_mse',
            'max_features': None,
            'max_depth': 15,
            'warm_start': False
            }

        gb = GradientBoostingClassifier(**best_parameters_gb['f1'])
        gb.fit(X_train, y_train)
        models.append(gb)

    ### Create list of used model names as model_nms
    for model in models:
        model_nms.append(str(model).split('(')[0])

    ### Create StratifiedKFold generator
    cv = StratifiedKFold(y, n_folds=5, shuffle=True)

    #### Determine optimal number of features
    n = 15 # number of features to list
    for i, model in enumerate(models):
        rfecv = RFECV(estimator=model, step=1, cv=cv, scoring='f1')
        try:
            rfecv.fit(X_t, y)
        except:
            set_trace()
        print_file.write("\nOptimal number of features for %s: %d" % (model_nms[i], rfecv.n_features_))
        srt_idx = np.argsort(rfecv.grid_scores_)
        srted_scores = rfecv.grid_scores_[srt_idx]
        srted_columns = df.columns[srt_idx]
        print_file.write("\n\nTop %d Features:\n" % n)
        for j in range(n):
            print_file.write("%d). %s - %f\n" % (j+1, srted_columns[j], srted_scores[j]))
        ### Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.tight_layout()
        plt.savefig(plot_dir + 'nm_feat_plot_' + model_nms[i] + trial + '.png')
        plt.close()
        ### Plot feature importances
        max_bar = srted_scores[:n].max()
        plt.figure()
        plt.title("Importance of Each Feature for %s" % (model_nms[i]))
        plt.bar(range(n), np.subtract(max_bar, srted_scores[:n]), color="g", align="center")
        plt.xticks(range(n), srted_columns[:n], rotation=75)
        plt.ylabel("Importance")
        plt.xlim([-1, n])
        plt.tight_layout()
        plt.savefig(plot_dir + "feat_imp_" + model_nms[i]  + trial + ".png")
        plt.close()
        ### Plot histograms of data for top n features
        top_nm = 6
        plt.figure()
        # bars = np.array(df[srted_columns[:top_nm]]).T
        bars = srted_columns[:top_nm].tolist()
        # nm, bins, patches = plt.hist(bars, label=labels, bins=50, normed=1, alpha=0.75)
        # plt.legend(loc="upper right")
        df[bars].diff().hist(alpha=0.75, bins=50, grid=True, normed=1) #, log=True
        plt.title("Histograms of Top %d Features" % top_nm)
        plt.tight_layout()
        # plt.xlim([-0.05e9 , .2e9])
        # plt.axis([0.0, 2.0, 0.0, 1.10])
        # plt.grid(True)
        plt.savefig(plot_dir + "top_feat_hists_" + model_nms[i]  + trial + ".png")
        plt.close()

    ### Draw a confusion matrixes for the results
    use_prob=True
    threshold = .25
    y_prob = []
    y_pred = []
    for i, model in enumerate(models):
        y_prob.append(model.predict_proba(X_test)[:,1])
        if use_prob:
            threshold = .5
            y_pred.append((y_prob[i] > threshold).astype(float))
        else:
            y_pred.append(model.predict(X_test))
        print_file.write("\n%s: confusion matrix:\n" % model_nms[i])
        print_file.write(str(pd.crosstab(pd.Series(y_test), pd.Series(y_pred[i]), rownames=['True'], colnames=['Predicted'], margins=True)))
        print_file.write("\n")

    ### Plot histogram of default probabilities
    y_prob_use = []
    prob_threshold = .01
    for i in range(len(models)):
        y_prob_use.append(y_prob[i][y_prob[i]>prob_threshold])
        prob_dict = {model_nms[i]: y_prob_use[i]}
        df_prob = pd.DataFrame(prob_dict)
        plt.figure()
        df_prob.plot.hist(alpha=0.75, bins=50, grid=True)
        plt.title("Histogram of Probabilities for %s" % (model_nms[i]))
        plt.tight_layout()
        plt.savefig(plot_dir + "default_prob_hist_" + model_nms[i]  + trial + ".png")
        plt.close()

    print_file.write("\n       Model            |  Accuracy     |    Precision     |      Recall    |       F1")
    for i, model in enumerate(models):
        accuracy, precision, recall, f1_score = get_scores(lr, X_test, y_train, y_test, y_pred[i])
        print_file.write("\n    %s   %f           %f           %f           %f" % (model_nms[i], accuracy, precision, recall, f1_score))

    ### Use plot_roc function provided during random forests to visualize curve of each #model
    print_file.write("\nUse the `plot_roc_curve` function to visualize the roc curve: See files in 'plots'.\n")
    for model in models:
        plot_roc_curve(X=X, y=y, plot_dir=plot_dir, trial=trial, cv=cv, model=model)

    ### Print probability of particular loan_id
    check_id = 504139525
    for i, model in enumerate(models):
        id_prob = model.predict_proba(X_t[loan_ids == check_id][0])[:,1]
        print_file.write("\nUsing model: %s" % model_nms[i])
        print_file.write("\nProbability of default for Loan_ID: %d is %0.4g\n" % (check_id, id_prob))

    ### Print top 'm' loans that have highest default probability, but not currently flagged
    ## percent of loan balance recovered following foreclosure 0.6930486709
    ## percent of loan balance lost following foreclosure 0.3069513291
    m = 5
    loss_pct = 0.3069513291
    nondef = y==0
    X_nondef = X_t[nondef]
    loans_nondef = loan_ids[nondef]
    df_nondef = df[nondef]
    df_nondef = df_nondef.reset_index(drop=True)
    for i, model in enumerate(models):
        y_probs_nd = model.predict_proba(X_nondef)[:,1]
        srt_idx = np.argsort(y_probs_nd)[::-1]
        top_m_probs = y_probs_nd[srt_idx][:m]
        top_m_loan_ids = loans_nondef[srt_idx][:m]
        df_top_m = df_nondef.iloc[srt_idx]
        print_file.write("\nUsing model: %s" % model_nms[i])
        print_file.write("\n      Loan ID:  |  Balance:  |  Default Prob:  |  Potential Loss:  |")
        for j, loan in enumerate(top_m_loan_ids):
            print_file.write("\n%d.      %d         %s         %0.4f      %s" % (j+1, loan, '${:,.0f}'.format(df_top_m['current_balance'].iloc[j]), top_m_probs[j], '${:,.0f}'.format(df_top_m['current_balance'].iloc[j] * top_m_probs[j] * loss_pct)))
        tot_pot_loss = np.sum(df_nondef['current_balance']*y_probs_nd)*loss_pct
        tot_bal = df_nondef['current_balance'].sum()
        print_file.write("\n\nTotal outstanding balance for all loans not already in default: %s" % ('${:,.0f}'.format(tot_bal)))
        print_file.write("\nTotal potential loss for loans not already in default: %s (%s)" % ('${:,.0f}'.format(tot_pot_loss), "{0:.3f}%".format(float(tot_pot_loss)/float(tot_bal) * 100.)))

    print_file.close()
