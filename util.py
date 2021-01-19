import numpy
from collections import Counter
from datetime import date
from itertools import cycle
from os.path import exists

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import entropy, ks_2samp, wilcoxon

from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2,f_classif
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score, precision_score, recall_score, precision_recall_curve, \
    roc_curve, average_precision_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras import backend as K, models

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data_dir = 'APS\\datasets'
report_dir = 'reports/'
models_dir = 'models/'
generated_dir = 'generated_data/'
plot_dir = 'images/'
VAE_dir = 'VAE/'
GAN_dir = 'GAN/'
CGAN_dir = 'CGAN/'
VAE_CGAN_dir = 'VAE_CGAN/'
CGAN_VAE = 'CGAN_VAE/'

def classify_baseline_model(y_test):
    print("Majority class labels in the training data-set are negative i.e. 0 labels")

    test_pred_labels = np.zeros(len(y_test))
    accuracy_value = accuracy_score(y_test, test_pred_labels)
    f1_value = f1_score(y_test, test_pred_labels)
    confusion_base = confusion_matrix(y_test, test_pred_labels)
    print("***************************")
    print("The Accuracy on test set using baseline models is ", accuracy_value)
    print("The F1 Score on test set using baseline models is ", f1_value)
    print("The Confusion Matrix for baseline models is ", confusion_base)
    false_positive = confusion_base[0][1]
    false_negative = confusion_base[1][0]
    total_cost = 10 * false_positive + 500 * false_negative
    print("Total Cost using baseline models is ", total_cost)

## Load PCA to transform train and Test in PCA component
## With 98% of component
def load_PCA_data(train_data,test_data,data_dir,imputation_type):
    feature_list = list(set(train_data.columns) - set(['failure']))
    train_x = train_data.loc[:, feature_list]
    test_x = test_data.loc[:, feature_list]
    train_y = train_data.loc[:, 'failure']
    test_y = test_data.loc[:, 'failure']

    pca = PCA(0.98).fit(train_x)
    print("Number of principal componenets for 98% variance:::", pca.n_components_)

    columns = ['pca_%i' % i for i in range(pca.n_components_)]
    pca_train = pd.DataFrame(pca.transform(train_x), columns=columns, index=train_x.index)
    pca_test = pd.DataFrame(pca.transform(test_x), columns=columns, index=test_x.index)

    pca_train['failure'] = train_y
    pca_test['failure'] = test_y

    pca_train.to_csv(data_dir + 'pca_preprocessed_train-' + imputation_type + '.csv')
    pca_test.to_csv(data_dir + 'pca_preprocessed_test-' + imputation_type + '.csv')

    return pca_train,pca_test

#### Apply Feature Selection ############
#### Select K=120 feature using SelectKBest #######
def feature_selection(train,test,k_best=80):
    train_data = train.copy()
    test_data=test.copy()
    feature_list = list(set(train_data.columns) - set(['failure']))

    train_x = train_data.loc[:, feature_list]
    test_x = test_data.loc[:, feature_list]

    train_y = train_data.loc[:, 'failure']
    test_y = test_data.loc[:, 'failure']

    selectKBest = SelectKBest(f_classif, k_best)
    selectKBest.fit(train_x, train_y.values)
    best_train_features = selectKBest.transform(train_x)
    idxs_selected = selectKBest.get_support(indices=True)
    best_train_features = train_x.iloc[:, idxs_selected]
    best_test_features = test_x.iloc[:, idxs_selected]

    best_train_features['failure'] = train_y
    best_test_features['failure'] = test_y

    best_train_features.to_csv('datasets/select_feature-'+str(k_best)+'_train.csv')
    best_test_features.to_csv('datasets/select_feature-'+str(k_best)+'_test.csv')

    return best_train_features,best_test_features

# Load Preprocess Data
def load_preprocess_aps_data(imputation_type):
    train_data = pd.read_csv(
        r'datasets/train_processed_aps_'+imputation_type+'.csv',
        index_col=0)
    test_data = pd.read_csv(
        r'datasets/test_processed_aps_'+imputation_type+'.csv',
        index_col=0)

    train_data['failure'] = train_data.failure.replace(['neg','pos'],[0,1])
    test_data['failure'] = test_data.failure.replace(['neg','pos'],[0,1])
    print("======================Data summary==================")
    return train_data,test_data

## Return failure data with feature and without failure class
def failure_or_not(dftrain):
    failure = dftrain.loc[dftrain['failure'] == 1]
    print(failure.shape)
    return failure.drop('failure', axis=1)

## Split X and Y variable
def split_XY(train_data):
    X_train = train_data.loc[:, train_data.columns != 'failure'].copy()
    Y_train = train_data.loc[:, train_data.columns == 'failure'].copy()
    return  (X_train,Y_train)

# Split X and Y for train and Test sets
def load_failure_aps(train_data,test_data):

    train_data = train_data[train_data.failure==1]
    test_data = test_data[test_data.failure==1]

    X_train = train_data.loc[:, train_data.columns != 'failure']
    Y_train = train_data.loc[:, train_data.columns == 'failure']

    X_test = test_data.loc[:, test_data.columns != 'failure']
    Y_test = test_data.loc[:, test_data.columns == 'failure']

    return (X_train,Y_train),(X_test,Y_test)

""" Find the optimal probability cutoff point for a classification model related to event rate
    using roc_curve and G-Mean
"""
def roc_curve_threshold(clf, test_X, testy, yhat):

    fpr, tpr, thresholds = roc_curve(testy, yhat[:, 1])

    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    #print("====================" + str(thresholds[ix]) + "==============================")
    y_pred = clf.predict_proba(test_X)[:, 1] > thresholds[ix]
    print("====Classification report===========\n",classification_report(testy,y_pred))
    con_mat = confusion_matrix(testy, y_pred)
    tot_cost = con_mat[0][1] * 10 + con_mat[1][0] * 500
    print("-" * 117)
    print("Type 1 error (False Positive) = ", con_mat[0][1])
    print("Type 2 error (False Negative) = ", con_mat[1][0])
    print("-" * 117)
    print("Total cost = ", tot_cost)
    print("-" * 117)
    return thresholds[ix],tot_cost,con_mat

    #return list(roc_t['threshold'])

""" Find the optimal probability cutoff point for a classification model related to event rate
    using precision_recall and fscore
"""
def best_threshold_precision_recall(clf,test_X,testy,yhat,name,algo):
    # calculate roc curves
    precision, recall, thresholds = precision_recall_curve(testy, yhat)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)

    # plot the precision-recall curves
    failure = len(testy[testy == 1]) / len(testy)
    plt.plot([0, 1], [failure, failure], linestyle='--', label='failure')
    plt.plot(recall, precision, marker='.', label=algo)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(report_dir+plot_dir+"performance"+'/'+name+'_precision_recall.jpg')
    plt.close()
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    #print("====================" + str(thresholds[ix]) + "==============================")
    y_pred = clf.predict_proba(test_X)[:, 1] > thresholds[ix]
    con_mat = confusion_matrix(testy, y_pred)
    print("====Classification report===========\n",classification_report(testy,y_pred))

    tot_cost = con_mat[0][1] * 10 + con_mat[1][0] * 500
    print("-" * 117)
    print("Type 1 error (False Positive) = ", con_mat[0][1])
    print("Type 2 error (False Negative) = ", con_mat[1][0])
    print("-" * 117)
    print("Total cost = ", tot_cost)
    print("-" * 117)

    return thresholds[ix],tot_cost,con_mat


"""
Plot Precision recall curve with Total cost curve
"""
def compute_threshold(y_ts,pred_prob,name,method):
    trail = 1
    plot = []
    for x in range(0, trail):
        precision, recall, thresholds = precision_recall_curve(y_ts, pred_prob[:,1])
        thresholds = np.append(thresholds, 1)
        print("==============index,thresholds:::",x,len(thresholds))
        costs = []
        for threshold in thresholds:
            y_pred_thres = pred_prob[:,1] > threshold
            c = confusion_matrix(y_ts, y_pred_thres)
            cost = c[0, 1] * 10 + c[1, 0] * 500
            costs.append(cost)


        plot.append({'threshold': thresholds, 'precision': precision, 'recall': recall, 'costs': costs})
        df = pd.DataFrame(plot)
        #df.to_csv("reports/threshold_cost.csv")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        for x in plot:
            plt.plot(x['threshold'], x['precision'], 'r')
            plt.plot(x['threshold'], x['recall'], 'g')

        plt.legend(('precision', 'recall'))
        plt.xlabel('Threshold')
        plt.ylabel("Precision/Recall")
        plt.subplot(1, 2, 2)
        for x in plot:
            plt.plot(x['threshold'], x['costs'], 'y')
        plt.legend(('Total costs'))
        plt.xlabel('Threshold')
        plt.ylabel("Total Cost")
        plt.savefig(report_dir + '/images/performance/' +method +name + '_thresholds_cost.jpg')
        plt.close()

def total_cost(clf, test_X, y_true, y_pred, y_pred_prob, ks_test1, ks_test2, fid, algo,name='Baseline Model'):

    cm = confusion_matrix(y_true, y_pred).ravel()
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    total_cost = 10 * fp + 500 * fn
    print("======Performance using "+ name +" without threshold tunning ======= \n ")
    print("Total Cost :: ", total_cost)
    precision = precision_score(y_true, y_pred, average='binary')
    print('Precision: %.3f' % precision)
    recall = recall_score(y_true, y_pred, average='binary')
    print('Recall: %.3f' % recall)
    f1 = f1_score(y_true, y_pred, average='binary')
    print('f1_score: %.3f' % f1)

    print("\n======Performance using "+ name +" after threshold tunning ======= \n ")
    threshold_roc, threshold_cost_roc,cm = roc_curve_threshold(clf, test_X, y_true, y_pred_prob)
    threshold_f1score, threshold_cost_f1score,cm = best_threshold_precision_recall(clf, test_X, y_true, y_pred_prob[:, 1],name,algo)

    if threshold_cost_f1score < threshold_cost_roc:
        threshold, threshold_cost,cm_f1 = threshold_f1score, threshold_cost_f1score,cm
    else:
        threshold, threshold_cost,cm_roc = threshold_roc, threshold_cost_roc,cm


    time = date.today()
    if  exists(report_dir+'final.csv'):
        total_cost_df = pd.read_csv(report_dir+'final.csv',index_col=0)
        #print('Total cost history summary ::\n', total_cost_df.head())
    else:
        total_cost_df = pd.DataFrame(columns=['time','name','ks-test1,ks-test2','fid','false_positive','false_negative','Precision','Recall','f1_score','total_cost','threshold','threshold_cost'])

    #compute_cost = compute_threshold(y_true,y_pred_prob)

    total_cost_df =total_cost_df.append({'time':time,'name':name,'ks-test1':ks_test1,'ks-test2':ks_test2,'fid':fid,'false_positive': fp, 'false_negative': fn,
                                         'Precision':precision,'Recall':recall,'f1_score':f1,
                                        'total_cost': total_cost,'threshold':threshold,'threshold_cost':threshold_cost},ignore_index = True)
    total_cost_df.to_csv(report_dir+'final.csv')


"""
TSNE plot using:
    Real Failure
    Non_failure
    Generated samples
"""
def tsne_data_comparision(real_pos,real_neg,generated_data,name,vae_sample,method='CGAN'):
    g_failure_df = pd.DataFrame(generated_data,columns=real_neg.columns).sample(n=1000,random_state=0)
    fig, ax = plt.subplots(figsize=(15, 4))
    legend = []

    failure_df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])  # .loc[:,feauters]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(real_pos)
    failure_df['tsne-2d-one'] = tsne_results[:, 0]
    failure_df['tsne-2d-two'] = tsne_results[:, 1]
    plt.scatter(failure_df['tsne-2d-one'], failure_df['tsne-2d-two'] ,c='green')
    legend.append('Real Failure data')

    if method=='VAE_CGAN' or method=='CGAN_VAE':
        v_failure_df = pd.DataFrame(vae_sample, columns=real_neg.columns)#.sample(n=1000, random_state=123)
        X_v = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])  # .loc[:,feauters]
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
        tsne_results = tsne.fit_transform(v_failure_df)
        X_v['tsne-2d-one'] = tsne_results[:, 0]
        X_v['tsne-2d-two'] = tsne_results[:, 1]
        plt.scatter(X_v['tsne-2d-one'], X_v['tsne-2d-two'], c='blue')
        legend.append('VAE Generated Failure data')


    nonfailure_df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])  # .loc[:,feauters]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(real_neg)
    nonfailure_df['tsne-2d-one'] = tsne_results[:, 0]
    nonfailure_df['tsne-2d-two'] = tsne_results[:, 1]
    plt.scatter(nonfailure_df['tsne-2d-one'], nonfailure_df['tsne-2d-two'], c='red')
    legend.append('Real Non Failure data')

    X_g = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])  # .loc[:,feauters]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(g_failure_df)
    X_g['tsne-2d-one'] = tsne_results[:, 0]
    X_g['tsne-2d-two'] = tsne_results[:, 1]
    plt.scatter(X_g['tsne-2d-one'], X_g['tsne-2d-two'],c = 'cyan')
    legend.append('Generated Failure data')

    ax.set_xlabel('tsne-2d-one')
    ax.set_ylabel('tsne-2d-two')
    ax.legend(legend)
    print(report_dir+plot_dir+"TSNE/"+method+'/'+name+'_tsne.jpg')
    plt.savefig(report_dir+plot_dir+"TSNE/"+method+'/'+name+'_tsne.jpg')
    plt.clf()
    plt.close()
    #plt.show()

def tsne_plot(X,name='real'):

    X_df = pd.DataFrame(columns=['tsne-2d-one','tsne-2d-two'])#.loc[:,feauters]


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)

    X_df['tsne-2d-one'] = tsne_results[:,0]
    X_df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        #hue="class",
        #palette=sns.color_palette("muted", 10),
        data=X_df,
        legend="full",
        alpha=0.3
    )
    plt.show()
    #plt.savefig(report_dir+plot_dir+name+'_tsne.jpg')


"""
VAE loss histroy plot
"""
def plot_vae_losses(hist, name='loss_history'):
    print(hist.history)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('models loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(report_dir+plot_dir+"VAE/"+name+'_loss-plot.jpg')

# create a line plot of loss for the gan and save to file
def plot_gan_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist,name):
	# plot loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='discriminator')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
	# plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend()
	# save plot to file
    print(report_dir+plot_dir+"performance/"+name+'_history-plot.jpg')
    plt.savefig(report_dir+plot_dir+"performance/"+name+'_history-plot.jpg')
    plt.close()

def plot_cost(tot_cost,model_type,name):
    epochs = [x['epochs'] for x in tot_cost]
    total_cost = [x['total_cost'] for x in tot_cost]
    plt.plot(epochs, label='epochs')
    plt.plot(total_cost, label='Total COst')
    plt.legend()
	# save plot to file
    #print(report_dir+plot_dir+"performance/"+name+'_history-plot.jpg')
    plt.savefig("best/"+model_type+'-'+name+'_cost_hostory.jpg')
    plt.close()


"""
    Calculate cluster info and decide best cluster  
    using silhouette_score
"""
def cluster_info(train_data):
    algorithm = cluster.KMeans
    prev_label,prev_score=None,None
    cl=2
    for i in [2,3,4,5]:
        args, kwds = (), {'n_clusters': i, 'random_state': 0}
        labels = algorithm(*args, **kwds).fit_predict(train_data[train_data.columns.drop('failure')])
        if prev_label==None:
            prev_label = labels
        print(pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'], index=np.unique(labels)))
        score = silhouette_score(train_data[train_data.columns.drop('failure')].values, labels, metric='euclidean')
        print('cluster and silhouette_score===',i,score)
        if prev_label==None:
            prev_label = labels
            prev_score = score
        elif score > prev_score:
            prev_label = labels
            prev_score = score
            cl = i

    return labels,cl

"""
Combine Generated sample with Train datasets
"""
def get_combined_generated_real(g_samples,train_data,name):
    #print(g_samples[0,:])
    dfnew = pd.DataFrame(g_samples, columns=train_data.columns.drop('failure'))
    dfnew['failure'] = np.ones(len(g_samples), dtype=np.int)
    #dfnew.to_csv(name+".csv")
    augmented = pd.concat([train_data, dfnew], ignore_index=True).sample(frac=1)
    print('Generated Dataframe summary:: ')
    print(dfnew.head())
    ent = entropy(list(Counter(augmented['failure']).values()), base=2)
    print("\nEntropy:::",ent)
    dfnew.to_csv(generated_dir+name+'.csv')
    return augmented

def balance_combined(g_samples,train_data,name):

    number_samples = g_samples.shape[0]+2000
    idxs_pos = train_data[train_data.failure == 1].index
    idxs_neg = train_data[train_data.failure == 0].sample(n=number_samples, replace=False, random_state=0).index
    idxs_balanced = np.concatenate((idxs_pos, idxs_neg))
    train_sample = train_data.loc[idxs_balanced]
    #train_balanced = train_data.loc[idxs_balanced]

    dfnew = pd.DataFrame(g_samples, columns=train_data.columns.drop('failure'))
    dfnew['failure'] = np.ones(len(g_samples), dtype=np.int)
    #dfnew.to_csv(name + ".csv")
    augmented = pd.concat([train_sample, dfnew], ignore_index=True).sample(frac=1)
    print('Generated Dataframe summary:: ')
    print(augmented.head())
    return augmented

"""
    Generator Network Layers
    Functions to define the layers of the networks used in the 'define_models'
    functions in GAN
"""
def generator_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count, activation='relu',kernel_initializer='he_uniform')(x)  # 1
    x = layers.Dense(base_n_count*4, activation='relu')(x)  # 2
    x = layers.Dense(data_dim,activation='tanh')(x)
    return x

"""
    Generator Network Layers 
    Functions to define the layers of the networks used in the 'define_models'
    functions in CGAN
"""
def generator_network_w_label(x, labels, data_dim, base_n_count,g_hidden_size):
    x = layers.concatenate([x, labels])
    if g_hidden_size ==2:
        x = layers.Dense(base_n_count, activation='relu')(x)  # 1
        x = layers.Dense(base_n_count * 4, activation='relu')(x)  # 2
    elif g_hidden_size==3:
        x = layers.Dense(base_n_count, activation='relu')(x)  # 1
        x = layers.Dense(base_n_count * 2, activation='relu')(x)
        x = layers.Dense(base_n_count * 4, activation='relu')(x) # extra
    else:
        x = layers.Dense(base_n_count*4, activation='relu')(x)

    x = layers.Dense(data_dim, activation='tanh')(x)
    x = layers.concatenate([x, labels])
    return x

"""
    Discriminator Network Layer
    Functions to define the layers of the networks used in the 'define_models'
    functions in GAN, CGAN and VAE_CGAN
"""
def discriminator_network(x,d_hidden_size,hiddendim):
    if d_hidden_size ==2:
        x = layers.Dense(hiddendim * 2)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        #x = layers.Dropout(0.2)(x)

        x = layers.Dense(hiddendim)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        #x = layers.Dropout(0.2)(x)

    elif d_hidden_size==3:
        x = layers.Dense(hiddendim * 3, kernel_initializer='he_uniform')(x)
        #x = layers.LeakyReLU()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        #x = layers.Dropout(0.2)(x)

        x = layers.Dense(hiddendim * 2)(x)
        #x = layers.LeakyReLU()(x)

        x = layers.LeakyReLU(alpha=0.2)(x)
        #x = layers.Dropout(0.2)(x)

        #x = layers.Dense(hiddendim , kernel_initializer='he_uniform')(x)
        #x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dense(int(hiddendim) )(x)
        #x = layers.LeakyReLU()(x)

        x = layers.LeakyReLU(alpha=0.2)(x)
    else:
        x = layers.Dense(hiddendim*2,kernel_initializer='he_uniform')(x)
        #x = layers.LeakyReLU()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        #x = layers.Dense(50)(x)
        #x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x

def get_random_sample(data,which='neg'):
    if which=='pos':
        xf = data[data.failure == 1].drop('failure', axis=1).sample(n=995, random_state=123)
    else:
        xf = data[data.failure == 0].drop('failure', axis=1).sample(n=995, random_state=123)
    return xf

def compare_distribution(xf,generated_samples,name):
    #xf = train[train.failure==1].drop('failure', axis=1).sample(n=1000,random_state=123)
    g_sample = pd.DataFrame(generated_samples,columns=xf.columns)#.sample(n=1000,random_state=12)
    pca_real = pca_2_components_transformation(xf)
    pca_gen = pca_2_components_transformation(g_sample)

    # print("==========wilcoxon test==========")
    # stat, p = wilcoxon(pca_real[0], pca_gen[0])
    # print('stat=%.3f, p=%.3f' % (stat, p))
    # if p > 0.05:
    #     print('Probably the same distribution')
    # else:
    #     print('Probably different distributions')

    ks_result1 = ks_2samp(pca_real[0],pca_gen[0])
    ks_result2 = ks_2samp(pca_real[1], pca_gen[1])
    print(ks_result1,ks_result2)
    print("====================pvalue===:::",ks_result1[1])
    #ks_df = pd.read_csv('reports/k-s-test.csv',index_col=0)
    #ks_df = ks_df.append({'name': name, 'pvalue1': ks_result1[1], 'pvalue2': ks_result2[1]},ignore_index=True)
    #ks_df.to_csv('reports/k-s-test.csv')

    #If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis
    #  that the distributions of the two samples are the same.
    return ks_result1[1],ks_result2[1]

def compare_attributes_distribution(xf,generated_samples,name):
    #xf = train[train.failure==1].drop('failure', axis=1).sample(n=1000,random_state=123)
    len_sample = generated_samples.shape[0]
    g_sample = pd.DataFrame(generated_samples,columns=xf.columns).sample(frac=1)
    aa_000_g = g_sample.aa_000
    aa_000_r = xf.aa_000

    random_g = g_sample.bs_000#values[:,8]
    random_r = xf.bs_000#.values[:,8]


    ks_result1 = ks_2samp(aa_000_g,aa_000_r)
    ks_result2 = ks_2samp(random_g, random_r)
    print(ks_result1,ks_result2)
    print("pvalue===:::",ks_result1[1].round(4),ks_result2[1].round(4))
    #ks_df = pd.read_csv('reports/k-s-test.csv',index_col=0)
    #ks_df = ks_df.append({'name': name, 'pvalue1': ks_result1[1], 'pvalue2': ks_result2[1]},ignore_index=True)
    #ks_df.to_csv('reports/k-s-test.csv')

    #If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis
    #  that the distributions of the two samples are the same.
    return ks_result1[1],ks_result2[1]


# calculate frechet inception distance
# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

"""
# Calculate FID score using Inception model
#https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
"""
def calculate_inception_fid(images1,images2):
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(162,))
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + numpy.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

#'MMD functions'
def compute_kernel(x, y):
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

def pca_2_components_transformation(data):
    # Principal Component Analysis
    pca = PCA(0.99).fit(data)
    pca_transform = pca.transform(data)
    #print("Number of features after PCA = ", pca.shape)
    return pca_transform

#Function to plot confusion matrix and find the total cost
def result(cm,method,name):
    sns.set(rc={'figure.figsize':(5,5)})
    class_label = ["negative", "positive"]
    df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
    sns.heatmap(df_cm, annot = True, fmt = "d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(report_dir+plot_dir+"performance/"+method+'/'+name+'_cunfusion_matrix.jpg')

    #print("Number of False Positive = ", cm[0][1])
    #print("Number of False Negative = ", cm[1][0])
    #Total_cost = cm[0][1] * 10 + cm[1][0] * 500
    #print("Total cost = ",Total_cost )
    #return Total_cost

def store_model(model,name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+"_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+"_model.h5")
    print("Saved model to disk")

def get_model(name):
    # load json and create model
    json_file = open(name+'_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("===Loaded model summary======\n")
    print(loaded_model.summary())

    # load weights into new model
    loaded_model.load_weights(name+"_model.h5")
    print("Loaded model from disk")
    return loaded_model

### Remove correlated from Train and Test sets
def remove_correlated_features(train,test):
    corr_matrix = train.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.8)]
    train_imp_features = train.drop(train[to_drop], axis=1)
    test_imp_features = test.drop(test[to_drop], axis=1)

    return train_imp_features,test_imp_features
