import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, f1_score, log_loss, roc_curve
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

from APS.app.APS_LJMU.util import total_cost, best_threshold_precision_recall, roc_curve_threshold, compute_threshold


def k_s_Test(name, original, generated):
    failure_real = original[original.failure == 1]
    failure_real = failure_real[failure_real.columns.drop('failure')]
    print(failure_real.shape, generated.shape)
    pca = PCA(n_components=2).fit(failure_real.values)
    transformed_failure_real = pca.transform(failure_real.values)
    pca = PCA(n_components=2).fit(generated[:, :-1])
    transformed_generated = pca.transform(generated[:, :-1])
    index = np.random.randint(transformed_generated.shape[0], size=failure_real.shape[0])
    # print("KS-Test index:", index)
    transformed_generated = transformed_generated[index, :]
    print(transformed_generated.shape, transformed_failure_real.shape)
    # print(transformed_failure_real[:,0])
    # print(transformed_generated[:,0])
    result1 = ks_2samp(transformed_failure_real[:, 0], transformed_generated[:, 0])
    result2 = ks_2samp(transformed_failure_real[:, 1], transformed_generated[:, 1])
    print(name)
    # stat1 = result1[0]
    # pval1= result1[1]
    print("\nk_s_Test===", result1, result2)


def performance_check(train, test_data, g_z, generator_name, epochs):
    print("\n\n======================Performance Check :: " + str(epochs) + "============\n")
    g_z_df = pd.DataFrame(g_z[:, :-1], columns=train.columns.drop('failure'))
    g_z_df['failure'] = np.ones((g_z_df.shape[0], 1))
    combined_train_df = pd.concat([train, g_z_df]).sample(frac=1)
    # print(train.shape,combined_train_df.shape,combined_train_df.failure.value_counts())
    X_train = combined_train_df[combined_train_df.columns.drop('failure')].values
    y_train = combined_train_df.failure

    X_test = test_data[test_data.columns.drop('failure')].values
    y_test = test_data.failure

    xgb = XGBClassifier(max_depth=9, n_estimators=27, n_jobs=-1, scale_pos_weight=40, min_child_weight=44)
    xgb = xgb.fit(X_train, y_train)
    # y_pred = xgb.predict(X_test)
    y_pred_prob = xgb.predict_proba(X_test)
    threshold_roc, threshold_cost_roc, con_mat = roc_curve_threshold(xgb, X_test, y_test, y_pred_prob)

    # tn, fp, fn, tp = con_mat.ravel()
    # total_cost = 10 * fp + 500 * fn
    print("\nTotal Cost using :: " + str(epochs) + " is ", threshold_cost_roc)
    print("=================Performance check End =====================\n")
    return threshold_cost_roc, combined_train_df
    # total_cost(xgb, y_test, y_pred,y_pred_prob,ks_test1,ks_test2,fid, str(epochs) + '::' + generator_name)


def multiple_classifier(X, y, Xt, yt, generator_name, ks_test1, ks_test2, fid):
    depth = list(range(6, 30, 3))
    nEstimator = list(range(20, 100, 30))
    tuned_parameters = [{'max_depth': depth, 'n_estimators': nEstimator}]

    # if no need of Gridsearch Training
    classifiers = {
        "XGBClassifier": XGBClassifier(max_depth=9, n_estimators=27, n_jobs=-1, scale_pos_weight=40,
                                       min_child_weight=44),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=6, max_depth=15, n_jobs=-1,
                                                        class_weight='balanced')
    }

    # classifiers = {
    #     "XGBClassifier": XGBClassifier(scale_pos_weight=40,min_child_weight=44),
    #     "RandomForestClassifier": RandomForestClassifier(class_weight='balanced',criterion='gini')
    # }
    for key, algo in classifiers.items():
        # gsv = GridSearchCV(algo, tuned_parameters, cv=5, verbose=1, scoring='roc_auc', return_train_score=True,n_jobs=-1)
        # gsv.fit(X, y)
        # print("Best HyperParameter: ", gsv.best_params_)
        # classifier = gsv.best_estimator_
        classifier = algo
        classifier.fit(X, y)
        pickle.dump(classifier, open(key, 'wb'))
        print(classifier.score(X, y))
        print(classifier.score(Xt, yt))
        y_pred = classifier.predict(Xt)
        report = classification_report(yt, y_pred)
        y_pred_prob = classifier.predict_proba(Xt)
        print(average_precision_score(yt, y_pred))
        total_cost(classifier, Xt, yt, y_pred, y_pred_prob, ks_test1, ks_test2, fid, key, key + '-' + generator_name)


def xgb_classifier(train_X, y_train, test_X, y_test, generator_name):
    # models fitting and hyper parameter tuning to find the best parameter
    print("-" * 117)
    print("====================XGB=======================")
    depth = list(range(3, 10, 3))
    nEstimator = list(range(15, 100, 10))
    tuned_parameters = [{'max_depth': depth, 'n_estimators': nEstimator}]

    # xgb = XGBClassifier(scale_pos_weight=59)
    # clf = GridSearchCV(xgb, tuned_parameters, cv=5, verbose=1, scoring='roc_auc', return_train_score=True, n_jobs=-1)
    # clf.fit(train_X, y_train)
    # print("Best HyperParameter: ", clf.best_params_)

    # models fitting using the best hyper parameter and predicting the cost
    clf = XGBClassifier(max_depth=9, n_estimators=27, n_jobs=-1, scale_pos_weight=40,
                                       min_child_weight=44)
    clf.fit(train_X, y_train)
    y_pred = clf.predict(test_X)

    print("classification_report\n", classification_report(y_test, y_pred))
    y_pred_prob = clf.predict_proba(test_X)

    threshold_roc, threshold_cost_roc, cm_roc = roc_curve_threshold(clf, test_X, y_test, y_pred_prob)
    threshold_f1score, threshold_cost_f1score, cm_f1 = best_threshold_precision_recall(clf, test_X, y_test,
                                                                                       y_pred_prob[:, 1],
                                                                                       generator_name, 'XGBClasifier')
    compute_threshold(y_test, y_pred_prob, generator_name, 'XGB')

    if threshold_cost_f1score < threshold_cost_roc:
        return threshold_f1score, threshold_cost_f1score, cm_f1
    else:
        return threshold_roc, threshold_cost_roc, cm_roc

    #total_cost(y_test, y_pred, 'xgb_classifier::' + '::' + generator_name)


def rf_classifier(train_X, y_train, test_X, y_test, generator_name):
    # models fitting and hyper parameter tuning to find the best parameter
    print("-" * 117)
    print("====================Random forest=======================")

    estimators = list(range(15, 100, 10))
    params = [{'criterion': ['gini'], 'max_features': ['sqrt'], 'n_estimators': estimators,
               'max_depth': [3, 6,9]}]
    rfc = RandomForestClassifier(random_state=0)
    # Executa grid search com cross validation
    # gsv = GridSearchCV(rfc, params, cv=5, scoring='precision', verbose=10, n_jobs=-1)
    # gsv.fit(train_X, y_train)
    # print("Best HyperParameter: ", gsv.best_params_)
    # clf = gsv.best_estimator_

    clf = RandomForestClassifier(criterion = 'gini',n_estimators=75, max_depth=9, n_jobs=-1,
                          class_weight='balanced')
    #models fitting using the best hyper parameter and predicting the cost
    clf.fit(train_X, y_train)

    y_pred = clf.predict(test_X)
    y_pred_prob = clf.predict_proba(test_X)
    report = classification_report(y_test, y_pred)
    print(evaluate(y_test, y_pred, y_pred_prob))
    # roc_curve_plot(y_test,y_pred)
    # ========================================================
    # compute_threshold(clf,train_X,y_train)

    threshold_roc, threshold_cost_roc, cm_roc = roc_curve_threshold(clf, test_X, y_test, y_pred_prob)

    threshold_f1score, threshold_cost_f1score, cm_f1 = best_threshold_precision_recall(clf, test_X, y_test,
                                                                                       y_pred_prob[:, 1],
                                                                                       generator_name, 'RFClassifier')
    compute_threshold(y_test,y_pred_prob,generator_name,'RF')

    if threshold_cost_f1score < threshold_cost_roc:
        return threshold_f1score, threshold_cost_f1score, cm_f1
    else:
        return threshold_roc, threshold_cost_roc, cm_roc

    # ===============================================================
    # compute_threshold(clf,train_X,y_train)


def evaluate(y_test, y_pred, y_pred_proba):
    if len(y_pred) > 0:
        f1 = f1_score(y_test, y_pred, average="weighted")
        print("F1 score: ", f1)
    if len(y_pred_proba) > 0:
        logloss = log_loss(y_test, y_pred_proba, eps=1e-15, normalize=True, sample_weight=None, labels=None)
        print("Log loss for predicted probabilities:", logloss)


def roc_curve_plot(Y_test, Y_pred):
    from sklearn.metrics import auc
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
