"""

RUN MODEL
 BASED ON ALL FEATURES
 ALL TRAINED MODEL STORE IN PRE_TRAINED/MODEL
 ALL OVERSAMPLE DATA STORE IN PRE_TRAINED/DATA
 ALL REPORT SAVE IN REPORTS DIR
"""

import pandas as pd

from APS.app.APS_LJMU.data_generator import BaseGenerator
from APS.app.APS_LJMU.util import get_model, compare_distribution, calculate_fid, feature_selection
from APS.app.APS_LJMU.classifier import multiple_classifier, xgb_classifier, rf_classifier
from APS.app.APS_LJMU.train_model import run_VanilaGANGenerator, run_VAEGenerator, run_CGANGenerator, run_VAECGAN_Generator, \
    run_CGAN_VAE_Generator
from APS.app.APS_LJMU.preprocess_aps import preprocess
from APS.app.APS_LJMU.util import load_preprocess_aps_data, get_random_sample, split_XY, get_combined_generated_real, \
    tsne_data_comparision, result

#train_main, test_main = load_preprocess_aps_data('mean')#
train_main, test_main = preprocess(imputation_type='mean')
base_generator = BaseGenerator(train_main,test_main)
n_samples = 5000
random_neg = get_random_sample(train_main,which='neg')
random_pos = get_random_sample(train_main,which='pos')

def vae_model():
    epochs_base, latent_size, name = 50, 64, "test_250_64_64_2_0005_VAE_no_mmd"
    n_layer = 2
    hidden_neurons = 64
    learning_rate = 0.0005
    vae_sample, ks_result1, ks_result2, fid = run_VAEGenerator(train_main,test_main,epochs_base,latent_size,n_samples,n_layer,hidden_neurons,learning_rate,name,random_neg)
    get_combined_generated = get_combined_generated_real(vae_sample, train_main, name)
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
                        fid)


def vanillaGAN_model():
    epochs, latent_size, name = 200, 50, "test_500-50-64-0001_Vanilla-GAN"
    learning_rate = 0.0005
    hidden_neurons = 64
    gan_sample, ks_result1, ks_result2, fid = run_VanilaGANGenerator(train_main, test_main, epochs, latent_size,
                                                                     hidden_neurons,n_samples, learning_rate,
                                                                     random_neg, random_pos,name)
    get_combined_generated = get_combined_generated_real(gan_sample, train_main, name)
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
                        fid)


def cgan_model():
    epochs, latent_size = 500, 50
    name = "best_with_cgan_best"
    n_layer = 3
    hidden_neurons = 128
    learning_rate = 0.0005
    cgan_sample, ks_result1, ks_result2, fid = run_CGANGenerator(train_main,test_main,epochs, latent_size,
                                                                 n_samples, n_layer, learning_rate,
                                                                 hidden_neurons, name, random_neg,random_pos)

    get_combined_generated = get_combined_generated_real(cgan_sample, train_main, name)
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
                        fid)

def vae_gan_model():
    epochs, latent_size = 1500, 50
    name = "best_test-2_vae_gan_gantrain_200-128_64_50_0005"
    n_layer = 3
    hidden_neurons = 128
    learning_rate = 0.0002

    cgan_sample,vae_sample, ks_result1, ks_result2, fid = run_VAECGAN_Generator(train_main, test_main, epochs,
                                                                                latent_size, n_samples, n_layer,
                                                                                learning_rate, hidden_neurons, name,
                                                                                random_neg,random_pos)
    print("VAE sample shape", vae_sample.shape)
    tsne_data_comparision(random_pos,random_neg,cgan_sample,name,vae_sample,method='VAE_CGAN')

    get_combined_generated = get_combined_generated_real(cgan_sample, train_main, name)

    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
                        fid)
    print("======================Both GAN and VAE=============================")
    name = name+"VAE_GAN-both"
    get_combined_generated = get_combined_generated_real(vae_sample, get_combined_generated, name)
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
    #                    fid)

def cgan_vae_model():
    epochs, latent_size = 200, 50
    name = "best_test_cgan-vae_100-128_64_50_0005"
    n_layer = 2
    hidden_neurons = 64
    learning_rate = 0.0005

    vae_sample,cgan_sample, ks_result1, ks_result2, fid = run_CGAN_VAE_Generator(train_main, test_main, epochs,
                                                                                 latent_size, n_samples, n_layer,
                                                                                 learning_rate, hidden_neurons, name,
                                                                                 random_neg,random_pos)

    tsne_data_comparision(random_pos,random_neg,vae_sample,name,cgan_sample,'CGAN_VAE')

    get_combined_generated = get_combined_generated_real(vae_sample, train_main, name)

    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
                        fid)
    print("======================Both GAN and VAE=============================")
    name = name+"CGAN-VAE-both"
    get_combined_generated = get_combined_generated_real(vae_sample, get_combined_generated, name)
    (train_X, train_y) = split_XY(get_combined_generated)
    (test_X, test_y) = split_XY(test_main)
    #multiple_classifier(train_X, train_y.values.ravel(), test_X, test_y.values.ravel(), name, ks_result1, ks_result2,
    #                    fid)


def run_with_pretrained_model(method,model_flag):

    if method == 'VAE':
        print("========================Start VAE==========================")
        name = 'classify-vae'

        if model_flag:
            vae = get_model('pre_trained/model/VAE')
            vae_sample = base_generator.generate_samples(vae,32,5000)
            #vae_sample = vae.generate_samples(5000)
            ks_result1, ks_result2 = compare_distribution(random_pos, vae_sample, name)
            fid = calculate_fid(random_pos, vae_sample)

            print("## VAE Quantative Analysis##")
            print("KS-Test 1 ", ks_result1)
            print("KS-Test 2 ", ks_result2)
            print(" FID ", fid)
            vae_combine = get_combined_generated_real(vae_sample, train_main, name)

        else:
            vae_sample = pd.read_csv(r'pre_trained/data/VAE.csv')
            vae_combine = get_combined_generated_real(vae_sample, train_main, name)
            #load combined data of train and VAE sample
            #vae_combine = pd.read_csv('pre_trained/data/vae_cgan_combine.csv',index_col=0)

        (train_X, train_y) = split_XY(vae_combine)
        (test_X, test_y) = split_XY(test_main)
        threshold, threshold_cost, cm = xgb_classifier(train_X, train_y, test_X, test_y, name)
        result(cm, 'VAE', 'XGB-classify-vae')
        threshold_rf, threshold_cost_rf, cm_rf = rf_classifier(train_X, train_y, test_X, test_y, name)
        result(cm_rf, 'VAE', 'RF-classify-vae')

        print("========================End VAE==========================")

    elif method == 'CGAN':
        print("\n========================Start CGAN==========================\n")
        name = 'classify-cgan'
        if model_flag:
            cgan = get_model('pre_trained/model/CGAN')
            cgan_sample = base_generator.generate_samples(cgan,64,5000)#cgan.generate_samples(5000)
            ks_result1, ks_result2 = compare_distribution(random_pos, cgan_sample, name)
            fid = calculate_fid(random_pos, cgan_sample)

            print("## CGAN Quantative Analysis##")
            print("KS-Test 1 ", ks_result1)
            print("KS-Test 2 ", ks_result2)
            print(" FID ", fid)

            cgan_combine = get_combined_generated_real(cgan_sample, train_main, name)

        else:
            #cgan_sample = pd.read_csv(r'pre_trained/data/cgan_800_50_64_2_0002_cluster_test.csv')
            #cgan_sample = pd.read_csv(r'generated_data/cgan_200_50_64_2_0002.csv')
            #cgan_combine = get_combined_generated_real(cgan_sample, train_main, name)

            #load combined data of CGAN sample and Train datsets
            cgan_combine = pd.read_csv(r'pre_trained/data/CGAN.csv',index_col=0)

        (train_X, train_y) = split_XY(cgan_combine)
        (test_X, test_y) = split_XY(test_main)
        test_X = test_X[train_X.columns]
        threshold, threshold_cost, cm = xgb_classifier(train_X, train_y, test_X, test_y, name)
        result(cm, 'CGAN', 'XGB-classify-vae')
        threshold_rf , threshold_cost_rf , cm_rf = rf_classifier(train_X, train_y, test_X, test_y, name)
        result(cm_rf, 'CGAN', 'RF-classify-cgan')
        print("\n========================End CGAN==========================")

    elif method == 'VAE_CGAN':
        print("\n========================Start VAE-CGAN==========================\n")
        name = 'classify-VAE_CGAN'

        if model_flag:
            #vae_sample = pd.read_csv(r'pre_trained/data/VAE_no_mmd.csv')
            vae = get_model('pre_trained/model/VAE')
            vae_sample = base_generator.generate_samples(vae,50,5000)#vae.generate_samples(5000)
            ks_result1, ks_result2 = compare_distribution(random_pos, vae_sample, name)
            fid = calculate_fid(random_pos, vae_sample)
            print("##VAE Quantative Analysis##")
            print("KS-Test 1 ",ks_result1)
            print("KS-Test 2 ",ks_result2)
            print(" FID ",fid)

            cgan = get_model('pre_trained/model/VAE_CGAN')
            cgan_sample = base_generator.generate_samples(cgan,32,5000)#cgan.generate_samples(5000)
            ks_result1, ks_result2 = compare_distribution(random_pos, cgan_sample, name)
            fid = calculate_fid(random_pos, cgan_sample)

            print("##VAE-CGAN Quantative Analysis##")
            print("KS-Test 1 ", ks_result1)
            print("KS-Test 2 ", ks_result2)
            print(" FID ", fid)

            vae_cgan_combine = get_combined_generated_real(cgan_sample, train_main, name)

        else:
            # cgan_sample = pd.read_csv(r'generated_data/cgan_200_50_64_2_0002.csv')
            # cgan_combine = get_combined_generated_real(cgan_sample, train_main, name)

            #load combined data of CGAN sample and Train datsets
            #vae_gan_gantrain_200_32_50_0002_test
            vae_cgan_combine = pd.read_csv(r'pre_trained/data/VAE_CGAN.csv',index_col=0)

        print("VAE CGAN combine shape::",vae_cgan_combine.shape)
        (train_X, train_y) = split_XY(vae_cgan_combine)
        (test_X, test_y) = split_XY(test_main)
        threshold, threshold_cost, cm = xgb_classifier(train_X, train_y, test_X, test_y, name)
        result(cm, 'VAE_CGAN', 'XGB-classify-VAE_CGAN')
        threshold_rf, threshold_cost_rf, cm_rf = rf_classifier(train_X, train_y, test_X, test_y, name)
        result(cm_rf, 'VAE_CGAN', 'RF-classify-VAE-CGAN')
        print("========================End VAE-CGAN==========================")

def train_classify(method):
    if method == 'VAE':
        name='classify-vae'
        sample = pd.read_csv(r'pre_trained/data/VAE_no_mmd.csv')
        #print(sample.shape)
        get_vae_combine = get_combined_generated_real(sample, train_main, name)
        (train_X, train_y) = split_XY(get_vae_combine)
        (test_X, test_y) = split_XY(test_main)
        threshold, threshold_cost, cm = xgb_classifier(train_X, train_y, test_X, test_y, name)
        result(cm,'VAE','classify-vae')

    elif method=='CGAN':
        name = 'classify-cgan'
        cgan_sample = pd.read_csv(r'pre_trained/data/cgan_800_50_64_2_0002_cluster_test.csv')
        #cgan_sample = pd.read_csv(r'generated_data/cgan_200_50_64_2_0002.csv')

        print(cgan_sample.shape)
        cgan_combine = get_combined_generated_real(cgan_sample, train_main, name)
        (train_X, train_y) = split_XY(cgan_combine)
        (test_X, test_y) = split_XY(test_main)
        #threshold, threshold_cost, cm = xgb_classifier(train_X, train_y, test_X, test_y, name)
        threshold, threshold_cost, cm = rf_classifier(train_X, train_y, test_X, test_y, name)

        result(cm, 'CGAN', 'rf_classify-cgan')

    elif method=='VAE_CGAN':
        name = 'classify-VAE_CGAN'
        cgan_sample = pd.read_csv(r'pre_trained/data/vae_gan_gantrain_200_32_50_0002_test.csv')
        #cgan_sample = pd.read_csv(r'generated_data/cgan_200_50_64_2_0002.csv')

        print(cgan_sample.shape)
        get_vae_combine = get_combined_generated_real(cgan_sample, train_main, name)
        (train_X, train_y) = split_XY(get_vae_combine)
        (test_X, test_y) = split_XY(test_main)
        threshold, threshold_cost, cm = xgb_classifier(train_X, train_y, test_X, test_y, name)
        result(cm, 'VAE_CGAN', 'classify-VAE_CGAN')



if __name__ == '__main__':
    #vae_model()
    #vanillaGAN_model()
    #cgan_model()
    #vae_gan_model()
    #cgan_vae_model()
    for method in ['VAE','CGAN','VAE_CGAN','VGAN']:
        run_with_pretrained_model(method,False)
        # train_classify('CGAN')
