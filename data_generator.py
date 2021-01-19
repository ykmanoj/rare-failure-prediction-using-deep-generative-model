from scipy.stats import wilcoxon, ks_2samp
from sklearn import cluster
import numpy as np
import pandas as pd
import random

from imblearn.over_sampling import SMOTE

from keras import backend as K, models
from keras import layers
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.layers.core import Lambda, Activation
from keras.losses import mse
from keras.optimizers import Adam
from keras.backend import concatenate
from sklearn.metrics import silhouette_score
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from APS.app.APS_LJMU.classifier import multiple_classifier, performance_check
from APS.app.APS_LJMU.util import generator_network, pca_2_components_transformation, \
    failure_or_not, plot_vae_losses, split_XY, generator_network_w_label, \
    discriminator_network, compute_mmd, plot_gan_history, plot_cost, get_random_sample, calculate_fid


class BaseGenerator():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.models_dir = 'models/'
        self.models_dir_VAE = 'models/VAE/'
        self.models_dir_GAN = 'models/GAN/'
        self.models_dir_VAEGAN = 'models/VAE_GAN/'
        self.report_dir = 'reports/'
        self.generated_dir = 'generated_data/'
        self.val_train,self.val_test=train_test_split(self.train_data,test_size=0.2,random_state=123)

        print(":::::::::::::::: Model Validation :::::::\n")
        print("Validation Class Counts in Train", self.val_train['failure'].value_counts())
        print("Validation Class Counts in Train", self.val_test['failure'].value_counts())

        self.random_neg = get_random_sample(self.train_data, which='neg')
        self.random_pos = get_random_sample(self.train_data, which='pos')

    def build_model(self):
        pass

    def train_model(self):
        pass

    def generate_samples(self,model, l_dim,n_samples):
        noise = np.random.uniform(-1, 1, (n_samples, l_dim))
        g_sample = model.predict(noise)
        # tsne_plot(g_sample, self.name)
        #dfnew = pd.DataFrame(g_sample, columns=self.train_data.columns.drop('failure'))
        #dfnew['failure'] = np.ones(len(g_sample), dtype=np.int)
        #augmented = pd.concat((self.train_data, dfnew), ignore_index=True).sample(frac=1)
        #print(augmented.shape, g_sample.shape)
        # augmented.to_csv(self.name+'.csv')
        return g_sample

    def compare_distribution(self,xf,generated_samples):
        # xf = train[train.failure==1].drop('failure', axis=1).sample(n=1000,random_state=123)
        g_sample = pd.DataFrame(generated_samples, columns=xf.columns)  # .sample(n=1000,random_state=12)
        pca_real = pca_2_components_transformation(xf)
        pca_gen = pca_2_components_transformation(g_sample)

        # print("==========wilcoxon test==========")
        # stat, p = wilcoxon(pca_real[0], pca_gen[0])
        # print('stat=%.3f, p=%.3f' % (stat, p))
        # if p > 0.05:
        #     print('Probably the same distribution')
        # else:
        #     print('Probably different distributions')

        ks_result1 = ks_2samp(pca_real[0], pca_gen[0])
        ks_result2 = ks_2samp(pca_real[1], pca_gen[1])
        print(ks_result1, ks_result2)
        print("====================pvalue===:::", ks_result1[1])
        # ks_df = pd.read_csv('reports/k-s-test.csv',index_col=0)
        # ks_df = ks_df.append({'name': name, 'pvalue1': ks_result1[1], 'pvalue2': ks_result2[1]},ignore_index=True)
        # ks_df.to_csv('reports/k-s-test.csv')

        # If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis
        #  that the distributions of the two samples are the same.
        return ks_result1[1], ks_result2[1]

        #If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis
        #  that the distributions of the two samples are the same.

    def store_model(self,model,name):
        # serialize model to JSON
        model_json = model.to_json()
        with open(name+"_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name+"_model.h5")
        print("Saved model to disk")

    def get_model(self,name):
        # load json and create model
        json_file = open(name+'_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(name+"_model.h5")
        print("Loaded model from disk")

    def evaluate_Model(self, e, tot_cost, model_type):
        g_z = self.generate_samples(5000)  # self.generator.predict([latent_samples,cluster_labels])
        ks_result1, ks_result2 = self.compare_distribution(self.random_pos, g_z[:, :-1], "evaluation")
        fid = calculate_fid(self.random_pos, g_z[:, :-1])
        cost, combined_train_df = performance_check(self.val_train, self.val_test, g_z, self.name, e + 1)
        if cost < 6500:
            combined_train_df.to_csv('best/Best_model-' + str(e) + '-cost-' + str(cost) + self.name + '.csv')
            self.store_model(self.generator, name="best/Best-model-cost-" + str(cost) + '-epochs-' + str(
                e) + '-' + model_type + self.name)
            plot_cost(tot_cost, model_type, self.name)
        return cost,ks_result1, ks_result2,fid,combined_train_df


class SMOTEGenerator(BaseGenerator):
    def __init__(self, train_data, test_data):
        super(SMOTEGenerator, self).__init__(train_data, test_data)

    def generate_samples(self, n_samples):
        X, y = split_XY(self.train_data)
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        return X, y


class VAEGenerator(BaseGenerator):

    def __init__(self, train_data, test_data, epochs, l_dim, name):
        super(VAEGenerator, self).__init__(train_data, test_data)
        self.epochs = epochs*5
        self.l_dim = l_dim
        self.encoder, self.decoder, self.vae = None, None, None
        self.Xf = failure_or_not(self.train_data)
        self.test_data = test_data
        self.test_X = failure_or_not(test_data)
        self.name = name
        self.model_type='VAE'

    def build_model(self,layer,hiddendim,l_rate):
        #self.name = self.name+'layer-'+str(layer)+'hidden-'+str(hiddendim)+'learning_rate-'+str(l_rate)
        latentdim = self.l_dim
        # build encoder (first step)
        inputs = Input(shape=(self.Xf.shape[1],), name='encoder_input')
        if layer==3:
            x = Dense(hiddendim*3, activation='relu')(inputs)
            #x = Dense(hiddendim, activation='relu')(x)
            x = Dense(hiddendim*2, activation='relu')(x)
            x = Dense(hiddendim, activation='relu')(x)
        else:
            x = Dense(hiddendim * 2, activation='relu')(inputs)
            x = Dense(hiddendim, activation='relu')(inputs)

        z_mean = Dense(latentdim, activation='linear', name='z_mean')(x)
        z_sd = Dense(latentdim, activation='linear', name='z_sd')(x)

        # implement reparametrization trick
        def sampling(args):
            z_mean, z_sd = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
            return z_mean + K.square(z_sd) * epsilon

        z = Lambda(sampling, output_shape=(latentdim,), name='z')([z_mean, z_sd])

        # build encoder (second step)
        self.encoder = Model(inputs, [z_mean, z_sd, z], name='encoder')

        # build decoder (first step)
        latent_inputs = Input(shape=(latentdim,), name='z_sampling')
        if layer==3:
            x = Dense(hiddendim, activation='relu')(latent_inputs)
            #x = Dense(hiddendim*4, activation='relu')(x)
            x = Dense(hiddendim*2, activation='relu')(x)
            x = Dense(int(hiddendim*2), activation='relu')(x)
        else:
            x = Dense(hiddendim , activation='relu')(x)
            x = Dense(hiddendim*2, activation='relu')(latent_inputs)

        outputs = Dense(self.Xf.shape[1], activation='relu')(x)

        # build decoder (second step)
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        # build VAE models
        outputs = self.decoder(self.encoder(inputs)[2])

        self.vae = Model(inputs, outputs, name='vae_model')

        # reconstruction loss
        reconstruction_loss = mse(inputs, outputs)

        # Kullback-Leibler divergence
        kl_loss = (K.square(z_mean) + K.square(z_sd) - K.log(K.square(z_sd)) - 1) / 2
        kl_loss = K.sum(kl_loss, axis=-1)
        #mmd_loss = compute_mmd(inputs,outputs)
        # loss function = reconstruction loss + Kullback-Leibler divergence
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')#Adam(lr=l_rate))
        print(self.vae.summary())

    def train_model(self):
        vae_hist = self.vae.fit(self.Xf, epochs=self.epochs, batch_size=256,
                                validation_data=(self.test_X, None),
                                )
        plot_vae_losses(vae_hist, self.name + 'vae_loss_history')
        #self.encoder.save(self.models_dir + self.name + '_VAE_encoder.h5')
        #self.decoder.save(self.models_dir + self.name + '_VAE_decoder.h5')
        self.store_model(self.decoder,name=self.models_dir_VAE+self.name)
        #self.vae.save(self.models_dir + self.name + '_VAE.h5')
        #self.store_model(self.vae,name=self.models_dir_VAE+self.name)
        return self.name

    def generate_samples(self, n_samples):
        #noise = np.random.normal(loc=0, scale=1, size=n_samples * self.l_dim).reshape(n_samples, self.l_dim)
        noise = np.random.uniform(-1, 1, (n_samples, self.l_dim))
        g_sample = self.decoder.predict(noise)
        # tsne_plot(g_sample, self.name)
        dfnew = pd.DataFrame(g_sample, columns=self.train_data.columns.drop('failure'))
        dfnew['failure'] = np.ones(len(g_sample), dtype=np.int)
        augmented = pd.concat((self.train_data, dfnew), ignore_index=True).sample(frac=1)
        print(self.name, augmented.shape, self.train_data.shape)
        #augmented.to_csv(self.name+'.csv')
        return g_sample,noise


class VanilaGANGenerator(BaseGenerator):
    def __init__(self, train_data, test_data, epochs, l_dim, name):
        print("========== VanilaGANGenerator =====================")
        super(VanilaGANGenerator, self).__init__(train_data, test_data)
        self.epochs = epochs
        self.xf = failure_or_not(self.train_data)
        self.gan, self.generator, self.discriminator = None, None, None
        self.latent_size = l_dim
        self.feature_dim = self.xf.shape[1]
        self.name = name

    def make_latent_samples(self, n_samples):
        return np.random.normal(loc=0, scale=1, size=(n_samples, self.latent_size))

    def define_models_GAN(self,d_learning_rate,base_n_count, type=None):

        generator_input_tensor = layers.Input(shape=(self.latent_size,))
        generated_failure_tensor = generator_network(generator_input_tensor, self.feature_dim, base_n_count)

        generated_or_real_failure_tensor = layers.Input(shape=(self.feature_dim,))

        if type == 'Wasserstein':
            discriminator_output = critic_network(generated_or_real_failure_tensor, self.feature_dim, base_n_count)
        else:
            discriminator_output = discriminator_network(generated_or_real_failure_tensor, self.feature_dim, base_n_count)

        self.discriminator = Model(inputs=generated_or_real_failure_tensor, outputs=discriminator_output,
                                   name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=d_learning_rate),metrics=['accuracy'])

        self.generator = Model(inputs=[generator_input_tensor], outputs=[generated_failure_tensor],
                               name='generator')
        frozen_discriminator = Model(inputs=generated_or_real_failure_tensor, outputs=discriminator_output)
        frozen_discriminator.trainable = False

        discriminator_trainable_weights = len(self.discriminator.trainable_weights)  # for asserts, below
        generator_trainable_weights = len(self.generator.trainable_weights)

        noise_input = Input(shape=(self.latent_size,))
        gen_failure = self.generator([noise_input])
        validity_output = frozen_discriminator(gen_failure)

        self.gan = Model(inputs=[noise_input], outputs=validity_output, name='combined')
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=d_learning_rate))

        assert (len(self.discriminator._collected_trainable_weights) == discriminator_trainable_weights)
        assert (len(self.gan._collected_trainable_weights) == generator_trainable_weights)

        print(self.generator.summary())
        print(self.discriminator.summary())
        print(self.gan.summary())
        return self.generator, self.discriminator, self.gan

    def make_trainable(self, model, trainable):
        for layer in model.layers:
            layer.trainable = trainable

    def make_labels(self, size):
        return np.ones([size, 1]), np.zeros([size, 1])

    def generate_samples(self, n_samples):
        # Generate data to fill failure data
        latent_samples = self.make_latent_samples(n_samples)
        generated_data = self.generator.predict(latent_samples)
        # generated_data = pd.DataFrame(generated_data)
        return generated_data

    def epoch_loss(self, e, g, d, c):
        # Saving weights and plotting images
        print('Step: {} of {}.'.format(e, self.epochs))
        K.set_learning_phase(0)  # 0 = test

        # loss summaries
        print('Losses: G, D Gen, D Real: {:.4f}, {:.4f}, {:.4f}'.format(c[-1], g[-1], d[-1]))
        print('D Real - D Gen: {:.4f}'.format(d[-1] - g[-1]))
        print('Generator models loss: {}.'.format(c[-1]))
        print('Discriminator models loss gen: {}.'.format(g[-1]))
        print('Discriminator models loss real: {}.'.format(d[-1]))

    def save_model_weights(self):
        self.generator.save_weights(self.models_dir+self.name + '_generator.h5')
        self.discriminator.save_weights(self.models_dir+self.name + '_discriminator.h5')
        self.gan.save_weights(self.models_dir+self.name + '_combined_gan.h5')

    def train_model(self):

        batch_size = 128  # train batch size
        eval_size = 32  # evaluate size
        smooth = 0.1
        model_type = 'GAN'

        # Make labels for the batch size and the test size
        y_train_real, y_train_fake = self.make_labels(batch_size)
        y_eval_real, y_eval_fake = self.make_labels(eval_size)

        losses, d_r_losses, d_g_losses, compound_losses,real_acc,fake_acc,tot_cost =[], [], [], [], [],[],[]

        for e in range(1, self.epochs):
            for i in range(len(self.xf.values) // batch_size):
                # Real data (minority class)
                X_batch_real = self.xf.values[i * batch_size:(i + 1) * batch_size]

                # Latent samples
                latent_samples = self.make_latent_samples(batch_size)
                # Fake data (on minibatches)
                X_batch_fake = self.generator.predict_on_batch(latent_samples)

                # Train the discriminator
                #self.make_trainable(self.discriminator, True)
                #self.discriminator.trainable=True
                d_r_loss,d_a_r = self.discriminator.train_on_batch(X_batch_real, y_train_real )#* (1 - smooth))
                d_g_loss,d_a_g = self.discriminator.train_on_batch(X_batch_fake, y_train_fake)

                # Train the generator (the discriminator is fixed)
                self.make_trainable(self.discriminator, False)
                #self.gan.compile(optimizer=Adam(lr=g_learning_rate), loss='binary_crossentropy')
                compound_loss = self.gan.train_on_batch(latent_samples, y_train_real)
                d_g_losses.append(d_g_loss)
                d_r_losses.append(d_r_loss)
                compound_losses.append(compound_loss)
                real_acc.append(d_a_r)
                fake_acc.append(d_a_g)

            if e % 32 == 0:
                # Evaluate
                X_eval_real = self.xf.values[np.random.choice(len(self.xf.values), eval_size, replace=False)]

                latent_samples = self.make_latent_samples(eval_size)
                X_eval_fake = self.generator.predict_on_batch(latent_samples)

                d_loss, d_a_acc = self.discriminator.test_on_batch(X_eval_real, y_eval_real)
                d_fake, d_f_acc = self.discriminator.test_on_batch(X_eval_fake, y_eval_fake)
                g_loss = self.gan.test_on_batch(latent_samples, y_eval_real)

                cost,augumented_df = self.evaluate_Model(e, tot_cost, model_type)
                tot_cost.append({'epochs': e, 'name': str(e) + '-' + model_type, 'total_cost': cost})
                if cost < 6500:
                    augumented_df.to_csv('best/Best_model-' + str(e) + '-cost-' + str(cost) + self.name + '.csv')
                    self.store_model(self.generator, name="best/Best-model-cost-" + str(cost) + '-epochs-' + str(
                            e) + '-' + model_type + self.name)
                plot_cost(tot_cost, model_type, self.name)

                losses.append((d_loss,d_fake, g_loss))
                print("Epoch: {:>3}/{} Discriminator real Loss: {:>6.4f} "
                      "Discriminator Fake Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
                        e + 1, self.epochs, d_loss, d_fake, g_loss))


        plot_gan_history(d_r_losses, d_g_losses, compound_losses, real_acc, fake_acc,self.name)

        self.store_model(self.gan,name=self.models_dir_GAN+self.name)


class CGANGenerator(BaseGenerator):
    def __init__(self, train_data, test_data, epochs, l_dim, name):
        super(CGANGenerator, self).__init__(train_data, test_data)
        self.epochs = epochs
        self.xf = failure_or_not(self.train_data)
        self.latent_size = l_dim
        self.c_label_dim = 1
        self.name = name
        self.feature_dim = self.xf.shape[1]
        self.gan, self.generator, self.discriminator = None, None, None

    def kmean_cluster(self):
        algorithm = cluster.KMeans
        prev_label, prev_score = None, None
        cl = 2
        for i in [2, 3]:
            args, kwds = (), {'n_clusters': i, 'random_state': 0}
            labels = algorithm(*args, **kwds).fit_predict(self.xf)

            #print(pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'],
             #                  index=np.unique(labels)))
            score = silhouette_score(self.xf.values, labels, metric='euclidean')
            print('cluster and silhouette_score===', i, score)
            if prev_label is None:
                prev_label = labels
                prev_score = score
            elif score > prev_score:
                prev_label = labels
                prev_score = score
                cl = i
				
        return prev_label, cl

    def make_latent_samples(self, n_samples, sample_size):
        return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

    def define_models_CGAN(self, g_hidden_size,
                           d_hidden_size,
                           g_learning_rate,
                           d_learning_rate,
                           noise_dim, hiddendim, type=None):

        data_dim = self.xf.shape[1]
        label_dim =1
        generator_input_tensor = layers.Input(shape=(noise_dim,))
        labels_tensor = layers.Input(shape=(label_dim,))  # updated for class
        generated_image_tensor = generator_network_w_label(generator_input_tensor,
                                                           labels_tensor, data_dim,
                                                           hiddendim,g_hidden_size)  # updated for class

        generated_or_real_image_tensor = layers.Input(shape=(data_dim + label_dim,))  # updated for class

        if type == 'Wasserstein':
            discriminator_output = critic_network(generated_or_real_image_tensor, data_dim + label_dim,
                                                  hiddendim)  # updated for class
        else:
            discriminator_output = discriminator_network(generated_or_real_image_tensor, #data_dim + label_dim,
                                                         d_hidden_size,hiddendim)  # updated for class

        self.discriminator = Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output,
                              name='discriminator')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=d_learning_rate),metrics=['accuracy'])

        self.generator = Model(inputs=[generator_input_tensor,labels_tensor], outputs=[generated_image_tensor], name='generator')
        frozen_discriminator = Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output)
        frozen_discriminator.trainable = False

        discriminator_trainable_weights = len(self.discriminator.trainable_weights)  # for asserts, below
        generator_trainable_weights = len(self.generator.trainable_weights)

        noise_input = Input(shape=(noise_dim,))
        gen_failure = self.generator([noise_input,labels_tensor])
        validity_output = frozen_discriminator(gen_failure)

        self.gan = Model(inputs=[noise_input,labels_tensor], outputs=validity_output, name='combined')
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=g_learning_rate,beta_1=0.5))

        assert (len(self.discriminator._collected_trainable_weights) == discriminator_trainable_weights)
        assert (len(self.gan._collected_trainable_weights) == generator_trainable_weights)
        #=====================================================

        print(self.generator.summary())
        print(self.discriminator.summary())
        print(self.gan.summary())
        return self.generator, self.discriminator, self.gan

    def make_trainable(self, model, trainable):
        for layer in model.layers:
            layer.trainable = trainable

    def make_labels(self, size):
        return np.ones([size, 1]), np.zeros([size, 1])

    def generate_samples(self, n_samples):
        # Generate data to fill failure data
        label_samples = np.array(random.choices([0, 1], k=n_samples))
        latent_samples = self.make_latent_samples(n_samples, self.latent_size)
        generated_data = self.generator.predict([latent_samples, label_samples])
        # generated_data = pd.DataFrame(generated_data)
        return generated_data

    def epoch_loss(self, e, g, d, c):
        # Saving weights and plotting images
        print('Step: {} of {}.'.format(e, self.epochs))
        K.set_learning_phase(0)  # 0 = test

        # loss summaries
        print('Losses: G, D Gen, D Real: {:.4f}, {:.4f}, {:.4f}'.format(c[-1], g[-1], d[-1]))
        print('D Real - D Gen: {:.4f}'.format(d[-1] - g[-1]))
        print('Generator models loss: {}.'.format(c[-1]))
        print('Discriminator models loss gen: {}.'.format(g[-1]))
        print('Discriminator models loss real: {}.'.format(d[-1]))

    def save_model_weights(self):
        self.generator.save_weights(self.models_dir+self.name + '_generator.h5')
        self.discriminator.save_weights(self.models_dir+self.name + '_discriminator.h5')
        self.gan.save_weights(self.models_dir+self.name + '_combined_gan.h5')

    def train_model(self,model_type):
        # Hyperparameters

        batch_size = 256  # train batch size
        eval_size = 256  # evaluate size
        smooth = 0.1

        # Make labels for the batch size and the test size
        y_train_real, y_train_fake = self.make_labels(batch_size)
        y_eval_real, y_eval_fake = self.make_labels(eval_size)
        cluster_labels,labels_dim = self.kmean_cluster()

        losses, d_r_losses, d_g_losses, compound_losses,real_acc,fake_acc,tot_cost = [], [],[], [], [], [],[]
        print("Number of batch::",len(self.xf.values) // batch_size)
        for e in range(self.epochs):
            for i in range(len(self.xf.values) // batch_size):
                # Real data (minority class)
                X_batch_real = self.xf.values[i * batch_size:(i + 1) * batch_size]
                X_batch_real_cluter_labels = cluster_labels[i * batch_size:(i + 1) * batch_size]
                # print(X_batch_real.shape,X_batch_real_cluter_labels.reshape(32,1).shape)
                X_batch_real = np.hstack((X_batch_real, X_batch_real_cluter_labels.reshape(batch_size, 1)))
                # Latent samples
                latent_samples = self.make_latent_samples(batch_size, self.latent_size)
                # Fake data (on minibatches)
                X_batch_fake = self.generator.predict_on_batch([latent_samples, X_batch_real_cluter_labels])

                # Train the discriminator
                #self.make_trainable(self.discriminator, True)
                #self.discriminator.compile()
                #X, y = np.vstack((X_batch_real, X_batch_fake)), np.vstack((y_train_real, y_train_fake))
                #d_l_r,d_a_r = self.discriminator.train_on_batch(X, y )

                d_l_r,d_a_r = self.discriminator.train_on_batch(X_batch_real, y_train_real )
                d_l_g,d_a_g = self.discriminator.train_on_batch(X_batch_fake, y_train_fake)
                # Train the generator (the discriminator is fixed)
                #self.make_trainable(self.discriminator, False)
                compound_loss = self.gan.train_on_batch([latent_samples, X_batch_real_cluter_labels], y_train_real)

                d_g_losses.append(d_l_g)
                d_r_losses.append(d_l_r)
                compound_losses.append(compound_loss)
                real_acc.append(d_a_r)
                fake_acc.append(d_a_g)

            # Evaluate
            if e % 10 == 0:
                evaluate_index = np.random.choice(len(self.xf.values), eval_size, replace=False)
                X_eval_real = self.xf.values[evaluate_index]
                X_eval_real_cluter_labels = cluster_labels[evaluate_index]
                X_eval_real = np.hstack((X_eval_real, X_eval_real_cluter_labels.reshape(eval_size, 1)))

                latent_samples = self.make_latent_samples(eval_size, self.latent_size)

                X_eval_fake = self.generator.predict_on_batch([latent_samples, X_eval_real_cluter_labels])

                d_loss,d_a_acc = self.discriminator.test_on_batch(X_eval_real, y_eval_real)
                d_fake,d_f_acc = self.discriminator.test_on_batch(X_eval_fake, y_eval_fake)
                g_loss = self.gan.test_on_batch([latent_samples, X_eval_real_cluter_labels], y_eval_real)
                if d_f_acc > 0.5 and d_a_acc > 0.5 and d_f_acc < 0.7\
                        and d_fake < 0.7 and d_fake > 0.5 \
                        and d_loss < 0.7 and d_loss > 0.5\
                        :
                    losses.append((d_loss, g_loss))
                    plot_gan_history(d_r_losses, d_g_losses, compound_losses, real_acc, fake_acc, self.name)
                    self.store_model(self.generator, name=model_type + self.name)

                    return self.name, self.generator

                cost, ks_result1, ks_result2, fid,augumented_df = self.evaluate_Model(e, tot_cost, model_type)
                tot_cost.append({'epochs': e, 'name': str(e) + '-' + model_type, 'total_cost': cost})
                if cost < 6500:
                    augumented_df.to_csv('best/Best_model-' + str(e) + '-cost-' + str(cost) + self.name + '.csv')
                    self.store_model(self.generator, name="best/Best-model-cost-" + str(cost) + '-epochs-' + str(
                            e) + '-' + model_type + self.name)
                plot_cost(tot_cost, model_type, self.name)

                losses.append((d_loss, g_loss))
                print("Epoch: {:>3}/{} Discriminator real Loss: {:>6.4f} "
                      "Discriminator Fake Loss: {:>6.4f} Generator Loss: {:>6.4f}"
                      "Real Accuracy: {:>6.4f}"
                      "Fake Accuracy: {:>6.4f}"
                      "XGB Cost: {:>6.4f}"
                      "KS 1: {:>6.4f}"
                      "FID score: {:>6.4f}"
                    .format(
                        e + 1, self.epochs, d_loss, d_fake, g_loss, d_a_acc, d_f_acc,
                        cost,ks_result1[1], fid))

        plot_gan_history(d_r_losses, d_g_losses, compound_losses, real_acc, fake_acc,self.name)
        self.store_model(self.generator,name=model_type+self.name)

        return self.name,self.generator



