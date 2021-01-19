import pandas as pd
import os
import gc

from scipy.stats import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from fancyimpute import KNN

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def preprocess(data_dir='datasets/',imputation_type='mean'):
    # X is the complete data matrix
    missing_gt = 60
    print(os.getcwd())
    aps_training_df = pd.read_csv(data_dir+'aps_failure_training_set.csv', error_bad_lines=False)
    aps_test_df = pd.read_csv(data_dir+'aps_failure_test_set.csv', error_bad_lines=False)
    print(aps_training_df.shape,aps_test_df.shape)
    print("=======Remove duplicates=================")
    aps_training_df.drop_duplicates(inplace=True)
    print(aps_training_df.shape,aps_test_df.shape)
    # aps_training_df = aps_training_df[aps_training_df['class']=='pos']
    print(aps_training_df.shape)
    print(aps_training_df.columns)
    #print(aps_training_df.isin(['na']).mean() * 100)
    print(aps_training_df.head())
    print('replacing na values to null=========')
    aps_training_df.replace(r'na', np.nan, regex=False, inplace=True)
    aps_test_df.replace(r'na', np.nan, regex=False, inplace=True)

    print('===removing more than '+str(missing_gt)+'% missing column=========')
    percent_missing = aps_training_df.isnull().sum() * 100 / len(aps_training_df)
    missing_value_df = pd.DataFrame({'column_name': aps_training_df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
    missing_gt_x = missing_value_df[missing_value_df['percent_missing'] > missing_gt].column_name

    print(missing_gt_x)
    aps_training_df = aps_training_df.drop(missing_gt_x, axis=1)
    aps_test_df = aps_test_df.drop(missing_gt_x, axis=1)

    print("============================Remove missing rows from Traning data==========================================")
    percent_missing_pos = (aps_training_df[aps_training_df['class'] == 'pos'].isnull().sum() / len(
    aps_training_df[aps_training_df['class'] == 'pos'])) * 100

    missing_value_df_pos = pd.DataFrame({'column_name': aps_training_df.columns,
                                         'percent_missing': percent_missing_pos})
    missing_value_df_pos.sort_values('percent_missing', inplace=True, ascending=False)
    prcent_row_missing = aps_training_df.isnull().sum(axis=1) * 100 / 170
    prcnt_50_row_missing = prcent_row_missing[prcent_row_missing > 50]
    print('More than 50% missing rows:::', len(prcnt_50_row_missing))
    #aps_training_df.drop(prcnt_50_row_missing.index, axis=0, inplace=True)
    print("============================Remove missing rows from Test data ==========================================")
    percent_missing_pos = (aps_test_df[aps_test_df['class'] == 'pos'].isnull().sum() / len(
            aps_test_df[aps_test_df['class'] == 'pos'])) * 100

    missing_value_df_pos = pd.DataFrame({'column_name': aps_test_df.columns,
                                         'percent_missing': percent_missing_pos})
    missing_value_df_pos.sort_values('percent_missing', inplace=True, ascending=False)
    prcent_row_missing = aps_test_df.isnull().sum(axis=1) * 100 / 170
    prcnt_50_row_missing = prcent_row_missing[prcent_row_missing > 50]
    print('More than 50% missing rows:::', len(prcnt_50_row_missing))
    #aps_test_df.drop(prcnt_50_row_missing.index, axis=0, inplace=True)

    #intersection_list = list(set(missing_value_df_pos.index) & set(prcnt_50_row_missing.index))
    #print("==Intersection list::::::::::",intersection_list)
    #aps_training_df.drop(intersection_list, axis=1, inplace=True)
    #aps_test_df.drop(intersection_list, axis=1, inplace=True)

    print("Training and Test data-set shape after dropping features is ", aps_training_df.shape, aps_test_df.shape)

    # Print number of positive classes and number of negative classes in the training data-set
    print("Number of positive classes = ", sum(aps_training_df['class'] == 'pos'))
    print("Number of negative classes = ", sum(aps_training_df['class'] == 'neg'))
    print("*******************")

    print("===================Drop outliers=================")
    from scipy import stats

    def drop_numerical_outliers(df, z_thresh=3):
        # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
        constrains = df.select_dtypes(include=[np.number]) \
            .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
            .all(axis=1)
        # Drop (inplace) values set to be rejected
        return df.drop(df.index[~constrains])

    # train = X_train
    # train['failure'] = Y_train

    #outlier = drop_numerical_outliers(aps_training_df)
    #print(outlier['class'].value_counts())
    #print(outlier.shape)

    # Extract features and labels from the training and test data-set
    y_train = aps_training_df[['class']].values
    x_train = aps_training_df.drop('class', axis=1)
    y_test = aps_test_df.loc[:, 'class'].values
    x_test = aps_test_df.drop('class', axis=1)

    columns = x_train.columns

    print('=======================remove duplicates====================')
    print('=============Missing Imputation=========')

    # Fill missing data in training and test data-set
    if(imputation_type=='median'):
        imputer = SimpleImputer(strategy='median')
        imputer.fit(x_train.values)
        x_train = imputer.transform(x_train.values)
        x_test = imputer.transform(x_test.values)

    elif (imputation_type == 'knn'):
        imputer = KNNImputer(n_neighbors=3)
        imputer.fit(x_train.values)
        x_train = imputer.transform(x_train.values)
        x_test = imputer.transform(x_test.values)

    elif (imputation_type == 'mean'):
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(x_train.values)
        x_train = imputer.transform(x_train.values)
        x_test = imputer.transform(x_test.values)

    else:
        x_train.fillna(-1,inplace=True)
        x_test.fillna(-1,inplace=True)


    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_df = pd.DataFrame(x_train,columns=columns)
    y_train_df = pd.DataFrame(y_train,columns=['failure'])

    x_test_df = pd.DataFrame(x_test, columns=columns)
    y_test_df = pd.DataFrame(y_test, columns=['failure'])

    #x_train_df, x_test_df = feature_selection(x_train_df,y_train_df,x_test_df)

    df_train_data = pd.concat([y_train_df, x_train_df], axis=1)
    df_train_data.to_csv("datasets/train_processed_aps_"+imputation_type+".csv")
    print(df_train_data.shape)

    df_test_data = pd.concat([y_test_df, x_test_df], axis=1)
    df_test_data.to_csv("datasets/test_processed_aps_"+imputation_type+".csv")
    print(df_test_data.shape)


    df_train_data['failure'] = df_train_data['failure'].replace(['neg', 'pos'], [0, 1])
    df_test_data['failure'] = df_test_data['failure'].replace(['neg', 'pos'], [0, 1])
    #gc.collect()
    print("Pre processing completed")

    return df_train_data,df_test_data


def class_preprocess(data_dir='datasets/', imputation_type='mean'):
    # X is the complete data matrix
    print(os.getcwd())
    aps_training_df = pd.read_csv(data_dir + 'aps_failure_training_set.csv', error_bad_lines=False)
    aps_test_df = pd.read_csv(data_dir + 'aps_failure_test_set.csv', error_bad_lines=False)

    # aps_training_df = aps_training_df[aps_training_df['class']=='pos']
    print(aps_training_df.shape)
    print(aps_training_df.columns)
    print(aps_training_df.isin(['na']).mean() * 100)
    print(aps_training_df.head())
    print('replacing na values to null=========')
    aps_training_df.replace(r'na', np.nan, regex=False, inplace=True)
    aps_test_df.replace(r'na', np.nan, regex=False, inplace=True)

    print('===removing more than 50% missing column=========')
    intersaction_list = ['ab_000', 'cr_000']
    aps_training_df.drop(intersaction_list, axis=1, inplace=True)
    aps_test_df.drop(intersaction_list, axis=1, inplace=True)
    #
    print("Training and Test data-set shape after dropping features is ", aps_training_df.shape, aps_test_df.shape)

    # Print number of positive classes and number of negative classes in the training data-set
    print("Number of positive classes = ", sum(aps_training_df['class'] == 'pos'))
    print("Number of negative classes = ", sum(aps_training_df['class'] == 'neg'))
    print("*******************")

    aps_training_df['failure_class'] = aps_training_df['class']
    aps_training_df.drop('class', axis=1, inplace=True)

    aps_test_df['failure_class'] = aps_test_df['class']
    aps_test_df.drop('class', axis=1, inplace=True)

    feature_list = aps_training_df.columns.drop('failure_class')
    failure_class = ['failure_class']

    print('=============Missing Imputation=========')

    # # Fill missing data in training and test data-set
    if (imputation_type == 'median'):
        imputer_neg = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer_pos = SimpleImputer(missing_values=np.nan, strategy='median')


    elif (imputation_type == 'knn'):
        imputer_neg = KNNImputer(n_neighbors=3)
        imputer_pos = KNNImputer(n_neighbors=3)

    elif (imputation_type == 'mean'):
        imputer_neg = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_pos = SimpleImputer(missing_values=np.nan, strategy='mean')

    X_pos = aps_training_df[aps_training_df['failure_class'] == 'pos']
    X_neg = aps_training_df[aps_training_df['failure_class'] == 'neg']

    X_pos = X_pos.loc[:, feature_list]
    X_neg = X_neg.loc[:, feature_list]

    X_neg = imputer_neg.fit_transform(X_neg)
    X_pos = imputer_pos.fit_transform(X_pos)

    X_pos_test = aps_test_df[aps_test_df['failure_class'] == 'pos']
    X_neg_test = aps_test_df[aps_test_df['failure_class'] == 'neg']
    X_pos_test = X_pos_test.loc[:, feature_list]
    X_neg_test = X_neg_test.loc[:, feature_list]

    X_neg_test = imputer_neg.transform(X_neg_test)
    X_pos_test = imputer_pos.transform(X_pos_test)

    pos_data = pd.DataFrame(X_pos, columns=feature_list)
    neg_data = pd.DataFrame(X_neg, columns=feature_list)
    pos_data['failure_class'] = np.ones((X_pos.shape[0], 1))
    neg_data['failure_class'] = np.zeros((X_neg.shape[0], 1))

    train_data = pd.concat([pos_data, neg_data])
    # Shuffle dataframe
    train_data = train_data.sample(frac=1)
    train_data.shape

    pos_data_test = pd.DataFrame(X_pos_test, columns=feature_list)
    neg_data_test = pd.DataFrame(X_neg_test, columns=feature_list)
    pos_data_test['failure_class'] = np.ones((X_pos_test.shape[0], 1))
    neg_data_test['failure_class'] = np.zeros((X_neg_test.shape[0], 1))

    test_data = pd.concat([pos_data_test, neg_data_test])
    # Shuffle dataframe
    test_data = test_data.sample(frac=1)
    test_data.shape

    #==================================================
    # Fill missing data in training and test data-set
    # if (imputation_type == 'median'):
    #     imputer = SimpleImputer(strategy='median')
    #
    # elif (imputation_type == 'knn'):
    #     imputer = KNNImputer(n_neighbors=3)
    #
    # elif (imputation_type == 'mean'):
    #     imputer = SimpleImputer(strategy='mean')
    #
    # columns = aps_training_df.columns
    #
    # train_data = aps_training_df.copy()
    # test_data = aps_test_df.copy()
    #
    # imputer.fit(train_data[feature_list].values)
    # train_data[feature_list] = imputer.transform(train_data[feature_list].values)
    # test_data[feature_list] = imputer.transform(test_data[feature_list].values)
    #
    # train_data['failure_class'] = train_data['failure_class'].replace(['neg', 'pos'], [0, 1])
    # test_data['failure_class'] = test_data['failure_class'].replace(['neg', 'pos'], [0, 1])

    #==================================================

    scaler = MinMaxScaler()
    # scaler = scaler.fit(train_data[feature_list])
    train_data[feature_list] = scaler.fit_transform(train_data[feature_list])
    test_data[feature_list] = scaler.transform(test_data[feature_list])

    train_data[feature_list] = train_data[feature_list].round(5)
    test_data[feature_list] = test_data[feature_list].round(5)
    test_data[feature_list].round(5).head()

    train_data['failure'] = train_data.failure_class  # .replace(['neg','pos'],[0,1])
    test_data['failure'] = test_data.failure_class  # .replace(['neg','pos'],[0,1])
    train_data.drop('failure_class', axis=1, inplace=True)
    test_data.drop('failure_class', axis=1, inplace=True)

    train_data.to_csv(data_dir+'class_final_preprocessed_train-' + imputation_type + '.csv')
    test_data.to_csv(data_dir+'class_final_preprocessed_test-' + imputation_type + '.csv')

    if (False):
        # Principal Component Analysis
        pca = PCA(0.98).fit(train_data[feature_list])
        print("Number of principal componenets for 98% variance:::",pca.n_components_)
        pca.fit(train_data[feature_list])
        train_data = pca.transform(train_data[feature_list])
        test_data = pca.transform(test_data[feature_list])
        print("Number of features after PCA = ", test_data.shape[1])

        train_data.to_csv(data_dir+'pca_preprocessed_train-' + imputation_type + '.csv')
        test_data.to_csv(data_dir+'pca_preprocessed_test-' + imputation_type + '.csv')

    # Save the data-sets to a csv file
    # gc.collect()
    print("Pre processing completed")

    return train_data, test_data

if __name__ == '__main__':
    preprocess()
