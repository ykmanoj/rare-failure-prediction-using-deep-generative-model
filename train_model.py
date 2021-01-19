import time

from APS.app.APS_LJMU.classifier import multiple_classifier, k_s_Test, xgb_classifier, rf_classifier
from APS.app.APS_LJMU.data_generator import VanilaGANGenerator, SMOTEGenerator, CGANGenerator, VAEGenerator
from APS.app.APS_LJMU.preprocess_aps import preprocess
from APS.app.APS_LJMU.util import load_failure_aps, classify_baseline_model, load_preprocess_aps_data, tsne_plot, split_XY, \
    get_combined_generated_real, load_PCA_data, tsne_data_comparision, \
    balance_combined, feature_selection, compare_distribution, calculate_fid, get_random_sample, \
    compare_attributes_distribution

import pandas as pd

"""
Run without sampling
"""
def run_without_sampling(train,test,id):
    name = id+'-Without-Sampling'
    (train_X, train_y)= split_XY(train)
    (test_X, test_y)= split_XY(test)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name)
    #xgb_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name)
    rf_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name)

"""
SMOTE oversampling
Classification
"""
def run_baseline_model(train,test,n_sample,name):
    sm_obj = SMOTEGenerator(train, test)
    train_X,train_y = sm_obj.generate_samples(n_sample)
    tsne_plot(train_X, "smoted_samples")
    (test_X, test_y) = split_XY(test)
    #classify_baseline_model(test_y)
    #tsne_data_comparision(train,gan_sample)
    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),'smote_train')

"""
VAE generator
    Training.
    TSNE plot after oversampling
    K-S Test and FID computation
    and return augumented datasets with KS and FID value
"""
def run_VAEGenerator(train,test,epochs,latent,n_samples,n_layer,
                     hidden_neurons,learning_rate,name,random_neg_sample):
    vae_obj = VAEGenerator(train, test, epochs, latent, name)
    vae_obj.build_model(n_layer,hidden_neurons,learning_rate)
    name = vae_obj.train_model()
    vae_sample,noise = vae_obj.generate_samples(n_samples)
    print(vae_sample.shape,train.shape)
    real_pos = train[train.failure == 1][train.columns.drop('failure')].sample(frac=1, random_state=123)
    ks_result1, ks_result2 = compare_distribution(real_pos, vae_sample, name)

    fid = calculate_fid(real_pos, vae_sample)
    print("Frechet Inception Distance:", fid)
    tsne_data_comparision(real_pos, random_neg_sample, vae_sample, name,'VAE')
    return vae_sample,ks_result1,ks_result2,fid

"""
Vanilla GAN generator
    Training.
    TSNE plot after oversampling
    K-S Test and FID computation
    and return augumented datasets with KS and FID value
"""
def run_VanilaGANGenerator(train, test, epochs, latent_size,hidden_neurons,
                           n_samples,learning_rate,random_neg_sample,real_pos,name):
    gan_obj = VanilaGANGenerator(train, test, epochs, latent_size ,name)
    gan_obj.define_models_GAN(learning_rate,hidden_neurons, type=None)
    gan_obj.train_model()
    gan_sample = gan_obj.generate_samples(n_samples)
    #tsne_plot(gan_sample, name)
    ks_result1, ks_result2 = compare_attributes_distribution(real_pos, gan_sample, name)
    fid = calculate_fid(real_pos, gan_sample)
    print("Frechet Inception Distance:", fid)
    tsne_data_comparision(real_pos, random_neg_sample, gan_sample, name,None,'VGAN')
    get_combined_generated = get_combined_generated_real(gan_sample, train, name)
    # print("Count of Failure and non failure",get_combined_generated.failure.value_counts())
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test)
    # classify_baseline_model(test_y)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
    #                    fid)
    # xgb_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name)
    # rf_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name)
    return gan_sample, ks_result1, ks_result2, fid


"""
Vanilla CGAN generator
    Training.
    TSNE plot after oversampling
    K-S Test and FID computation
    and return augumented datasets with KS and FID value
"""
def run_CGANGenerator(train, test, epochs, latent_size, n_samples, n_layer, learning_rate, hidden_neurons, name, random_neg_sample,real_pos,model_type='CGAN'):

    gan_obj = CGANGenerator(train, test, epochs, latent_size,name)
    gan_obj.define_models_CGAN(n_layer,n_layer,learning_rate,learning_rate,latent_size,hidden_neurons,type=None)
    gan_obj.train_model(model_type)
    gan_sample = gan_obj.generate_samples(n_samples)
    ks_result1, ks_result2 = compare_distribution(real_pos, gan_sample[:,:-1], name)
    fid = calculate_fid(real_pos,gan_sample[:,:-1])
    print("Frechet Inception Distance:",fid)
    if model_type=='CGAN':
        tsne_data_comparision(real_pos,random_neg_sample,gan_sample[:,:-1],name,None,model_type)
    
    #get_combined_generated = get_combined_generated_real(gan_sample[:,:-1],train,name)
    # print("Count of Failure and non failure",get_combined_generated.failure.value_counts())
    #(train_X, train_y) = split_XY(get_combined_generated)
    #(test_X, test_y) = split_XY(test)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name,ks_result1,ks_result2,fid)
    #xgb_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name)
    #rf_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name)
    return gan_sample[:,:-1],ks_result1,ks_result2,fid

"""
VAE-CGAN generator
    Training.
    TSNE plot after oversampling
    K-S Test and FID computation
    and return augumented datasets with KS and FID value
"""
def run_VAECGAN_Generator(train, test, epochs, latent_size, n_samples, n_layer,
                          learning_rate, hidden_neurons, name, random_neg_sample,random_pos):

    vae_sample = pd.read_csv(r'generated_data/VAE_no_mmd.csv').sample(n =1000,random_state=123)

    # vae_sample, ks_result1, ks_result2, fid = run_VAEGenerator(train, test, epochs, latent_size,
    #                                                            n_samples, n_layer,
    #                                                            hidden_neurons,
    #                                                            learning_rate, name,
    #                                                            random_neg)
    # # vae_sample, noise = vae_obj.generate_samples(n_samples)
    # get_vae_combine = get_combined_generated_real(vae_sample, train, "vae_gan")

    vae_combine = get_combined_generated_real(vae_sample, train, name)

    cgan_sample, ks_result3, ks_result4, fid2 = run_CGANGenerator(vae_combine, test, epochs, latent_size, n_samples,
                                                                  n_layer, learning_rate,
                                                                  hidden_neurons, name,
                                                                  random_neg_sample,random_pos,model_type='VAE_CGAN')
    tsne_data_comparision(random_pos, random_neg_sample,cgan_sample[:,:-1],name,vae_sample,'VAE_CGAN')

    #get_combined_generated = get_combined_generated_real(cgan_sample, train, name)
    return cgan_sample, vae_sample,ks_result3, ks_result4, fid2
    print("Count of Failure and non failure", get_combined_generated.shape,
          get_combined_generated.failure.value_counts())
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),
    #                    name, ks_result3, ks_result4, fid2)

    print("======================Both GAN and VAE=============================")
    name = name + "VAE_GAN-both"
    get_combined_generated = get_combined_generated_real(vae_sample, get_combined_generated, name)
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result3, ks_result4, fid2)

"""
CGAN-VAE generator
    Training.
    TSNE plot after oversampling
    K-S Test and FID computation
    and return augumented datasets with KS and FID value
"""
def run_CGAN_VAE_Generator(train_main, test_main, epochs_base, latent_size, n_samples, n_layer,
                          learning_rate, hidden_neurons, name, random_neg,random_pos):

    cgan_sample = pd.read_csv(r'generated_data/cgan_200_50_64_2.csv').sample(n =1000,random_state=123)

    # vae_sample, ks_result1, ks_result2, fid = run_VAEGenerator(train, test, epochs, latent_size,
    #                                                            n_samples, n_layer,
    #                                                            hidden_neurons,
    #                                                            learning_rate, name,
    #                                                            random_neg)
    # # vae_sample, noise = vae_obj.generate_samples(n_samples)
    # get_vae_combine = get_combined_generated_real(vae_sample, train, "vae_gan")

    print(cgan_sample.shape)
    cgan_sample_combine = get_combined_generated_real(cgan_sample, train_main, name)

    vae_sample ,ks_result1, ks_result2, fid = run_VAEGenerator(cgan_sample_combine, test_main, epochs_base, latent_size,
                                                               n_samples, n_layer, hidden_neurons, learning_rate, name,
                                                               random_neg)
    return vae_sample,cgan_sample, ks_result1,ks_result2,fid

"""
Train the VAE Model using
different model configuration
"""
def train_vae():
    datasets_dir = 'datasets/'
    imputation_types = ['mean','NA','knn']
    feature_engineerings = ['All','Features-selection','PCA']
    for imputation_type in imputation_types:
        train_main, test_main = preprocess(data_dir=datasets_dir,imputation_type=imputation_type)
        id = "APS-VAE" + imputation_type + "-"
        #train_main, test_main = load_preprocess_aps_data()
        for feature_engineering in feature_engineerings:
            if feature_engineering=='PCA':
                train,test = load_PCA_data(train_main,test_main,datasets_dir,imputation_type)
                feature_n = '-PCA='
            elif feature_engineering=='Features-selection':
                train,test = feature_selection(train_main,test_main)
                feature_n = '-Select_Features_80='
            else:
                feature_n = '-All_features='
                train,test = train_main,test_main

            random_neg = get_random_sample(train)
            #run_without_sampling(train,test,id)
            print(train.shape,test.shape)
            for i in [50, 100]:
                epochN = '-epochs='+str(i)
                for j in [32,64,128]:
                    latentN = '-latent_size=' + str(j)
                    for n_samples in [5000]:
                        n_sampleN = '-n_samples=' + str(n_samples)
                        for n_layer in [2,3]:
                            layer_N = '-n_layer=' + str(n_layer)
                            for hidden_neurons in [32, 64,128]:
                                hidden_N = '-hidden_neurons_base=' + str(hidden_neurons)
                                for learning_rate in [0.0005,0.0002]:
                                    lr_N = '-learning_rate=' + str(learning_rate)
                                    name = id+feature_n+epochN+latentN+n_sampleN+layer_N+hidden_N+lr_N
                                    vae_sample,ks_result1,ks_result2,fid = run_VAEGenerator(train,test,i,j,n_samples,n_layer,hidden_neurons,learning_rate,name,random_neg)
                                    get_combined_generated = get_combined_generated_real(vae_sample, train,name)
                                    (train_X, train_y) = split_XY(get_combined_generated)
                                    (test_X, test_y) = split_XY(test)
                                    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name,ks_result1,ks_result2,fid)

"""
Train the CGAN Model using
different model configuration
"""
def train_cgan():
    datasets_dir = 'datasets/'
    imputation_types = ['NA','mean','knn']
    feature_engineerings = ['Features-selection','ALL','PCA']
    for imputation_type in imputation_types:
        id = "APS-CGAN" + imputation_type + "-"
        train_main, test_main = preprocess(data_dir=datasets_dir,imputation_type=imputation_type)
        #train_main,test_main = load_preprocess_aps_data(imputation_type)
        for feature_engineering in feature_engineerings:
            if feature_engineering=='PCA':
                train,test = load_PCA_data(train_main,test_main,datasets_dir,imputation_type)
                feature_n = '-PCA='
            elif feature_engineering=='Features-selection':
                train,test = feature_selection(train_main,test_main,120)
                feature_n = '-Select_Features_K='+str(120)
            else:
                feature_n = '-All_features='
                train,test = train_main,test_main

            random_neg = get_random_sample(train,which='neg')
            random_pos = get_random_sample(train,which='pos')
            print(train.shape,test.shape)
            for i in [100, 300]:
                epochN = '-epochs='+str(i)
                for j in [32,64,128]:
                    latentN = '-latent_size=' + str(j)
                    for n_samples in [2000,5000]:
                        n_sampleN = '-n_samples=' + str(n_samples)
                        for n_layer in [2,3]:
                            layer_N = '-n_layer=' + str(n_layer)
                            for hidden_neurons in [32,64,128]:
                                hidden_N = '-hidden_neurons_base=' + str(hidden_neurons)
                                for learning_rate in [0.0005,0.0002]:
                                    lr_N = '-learning_rate=' + str(learning_rate)
                                    name = id+feature_n+epochN+latentN+n_sampleN+layer_N+hidden_N+lr_N
                                    #run_VanilaGANGenerator(train,test,i,j,n_samples,imputation_type,id)
                                    cgan_sample,ks_result1,ks_result2,fid = run_CGANGenerator(train, test, i, j, n_samples, n_layer, learning_rate, hidden_neurons, name, random_neg,random_pos)
                                    get_combined_generated = get_combined_generated_real(cgan_sample, train,name)
                                    (train_X, train_y) = split_XY(get_combined_generated)
                                    (test_X, test_y) = split_XY(test)
                                    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name,ks_result1,ks_result2,fid)

"""
Train the VAE-CGAN Model using
different model configuration
"""
def train_vae_cgan():
    datasets_dir = 'datasets/'
    imputation_types = ['mean', 'knn', 'NA']
    feature_engineerings = ['All','PCA','Features-selection']
    for imputation_type in imputation_types:
        id = "APS-VAE-CGAN_Both-" + imputation_type + "-"
        train_main, test_main = preprocess(data_dir=datasets_dir,imputation_type=imputation_type)
        #train_main, test_main = load_preprocess_aps_data(imputation_type)
        for feature_engineering in feature_engineerings:
            if feature_engineering == 'PCA':
                train, test = load_PCA_data(train_main, test_main, datasets_dir, imputation_type)
                feature_n = '-PCA-'
            elif feature_engineering == 'Features-selection':
                train, test = feature_selection(train_main, test_main,120)
                feature_n = '-Select_Features='+str(120)
            else:
                feature_n = '-All_features-'
                train, test = train_main, test_main

            random_neg = get_random_sample(train, which='neg')
            random_pos = get_random_sample(train, which='pos')
            print(train.shape, test.shape)
            for i in [200,400]:
                epochN = '-epochs=' + str(i)
                for j in [32,64,128]:
                    latentN = '-latent_size=' + str(j)
                    for n_samples in [2000, 5000]:
                        n_sampleN = '-n_samples=' + str(n_samples)
                        for n_layer in [2, 3]:

                            layer_N = '-n_layer=' + str(n_layer)
                            for hidden_neurons in [32, 64, 128]:
                                hidden_N = '-hidden_neurons_base=' + str(hidden_neurons)
                                for learning_rate in [0.0005,0.0002]:
                                    lr_N = '-learning_rate=' + str(learning_rate)
                                    name = id + feature_n + epochN + latentN + n_sampleN + layer_N + hidden_N + lr_N

                                    vae_sample, ks_result1, ks_result2, fid = run_VAEGenerator(train, test, i, j,
                                                                                               n_samples, n_layer,
                                                                                               hidden_neurons,
                                                                                               learning_rate, name,
                                                                                               random_neg)
                                    # vae_sample, noise = vae_obj.generate_samples(n_samples)
                                    get_vae_combine = get_combined_generated_real(vae_sample, train, "vae_gan")

                                    cgan_sample, ks_result3, ks_result4, fid2 = run_CGANGenerator(get_vae_combine, test, i, j, n_samples,
                                                                                                  n_layer, learning_rate,
                                                                                                  hidden_neurons, name,
                                                                                                  random_neg,random_pos)

                                    get_combined_generated = get_combined_generated_real(cgan_sample, train, name)
                                    print("Count of Failure and non failure", get_combined_generated.shape,
                                          get_combined_generated.failure.value_counts())
                                    (train_X, train_y) = split_XY(get_combined_generated)
                                    (test_X, test_y) = split_XY(test)
                                    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),
                                                        name, ks_result1, ks_result2, fid)

                                    # rf_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name)
                                    # xgb_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(),name)


##Train all the model###########
if __name__ == '__main__':
    print("######## Start Training #######")
    print("######## Start VAE Training #######")
    train_vae()
    print("######## Done VAE Training #######")
    print("######## Start CGAN Training #######")
    #train_cgan()
    print("######## Done CGAN Training #######")
    print("######## Start VAE-CGAN Training #######")
    #train_vae_cgan()
    print("######## Done VAE-CGAN Training #######")

