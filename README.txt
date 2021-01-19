################ LJMU ################################################
################ NAME: MANOJ KUMAR YADAV #############################
################ Project Name: RARE FAILURE PREDICTION USING DEEP GENERATIVE MODEL ON EXTREME CLASS IMBALANCE ################

1. train_model.py: Training multiple model with all experimented technique ##
2. run_model.py : Execute classifier with pre-trained oversampling model
and Data.
3. util.py : utility common file which get used in all module.
4. preprocess_aps.py : Preprocess steps included in this.
5. data_generator.py : All the oversampling model implemented here.
6. classifier.py : All classifier method implemented here.


#### Important to run ##############
Data Dir : Original datasets and preprocessed datasets
    datasets/
        aps_failure_training_set.csv
        aps_failure_test_set.csv

Model Dir : All model for each training steps
    models/
        CGAN/
        VAE/
        VAE_GAN/
        GAN/
Report Dir : Store all report in dir

    images/
        performance/
            GAN/
            VAE/
            CGAN/
            VAE_CGAN/
        TSNE/
            GAN/
            VAE/
            CGAN/
            VAE_CGAN/

Pre-Trained Dir : All pretrained model and agumented training datasets
    pretrained/
       data/
       model/

Generated Dir   :   Store all generated dataset
    generated_data/



