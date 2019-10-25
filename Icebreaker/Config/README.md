# Config explaination

* `Dataset_Settings`:
   * `test_size`: The proportion of the users that are used for creating test dataset
   * `missing_prop`: The proportion of missing data in dataset (used for creating artificial unobserved data in complete dataset)
* `Training_Settings`:
   * `flag_dataloader`: Set to True to use dataloader defined in the training file.
   * `batch_size`: Training batch size. At the beginning, the observed data number may be smaller than batch_size, then it will be automatically changed to half of the total observed data size. 
   * `z_sigma_prior`: Prior settings for latent space
   * `W_sigma_prior`: Prior settings for decoder 
   * `Drop_p`: Artificial missing proportion in each training step (equivalent to [EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE](https://arxiv.org/abs/1809.11142))
   * `flag_reset_optim`: Do not change, set it to True.
* `Optimizer`: Hyperparameters settings for Adam Optimizer
* `BNN_Settings`: Settings for model hyperparameters
   * `latent_dim`: The dimensionality of latent space
   * `dim_before_agg` and `embedding_dim`: Hyperparameters for partial encoder. For detailes refer to EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE](https://arxiv.org/abs/1809.11142)
   * `KL_coef`: The coefficient of KL penalty term in ELBO
   * `flag_log_q`: Whether to use log tricks to force positive sigma output. Otherwise, it will use the square trick. 
   * `encoder_settings`: The hyperparameters for partial encoder. Please refer to [EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE](https://arxiv.org/abs/1809.11142)
   * `decoder_layer_num`: The hidden layer number for decoder
   * `decoder_hidden`: Corresponding hidden unit number for decoder
   * `output_const`: Fixed multiplication constant at the decoder output (1 represents raw output, 2 represents doubling the raw output)
   * `add_const`: Fixed offset added at the decoder output
   * `sample_W`: Number of samples drawn for decoder weight
   * `init_range`: Initialization range for decoder weight
   * `coef_sample`: Always set to 0, this is to disable the BNN feature for baseline PNP model. 
* `Active_Learning_Settings`: 
   * `step`: Number of data points to select in active acquisition
   * `max_selection`: The maximum selected data number for active acquisition
   * `flag_clear_target_train`: Set False for imputation and True for active prediction
   * `flag_clear_target_test`: Same as above
   * `flag_hybrid`: Set to True to enable hybrid ELBO (refer to [Learning Structured Output Representation using Deep Conditional Generative Models](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf))
   * `balance_coef`: The balance proportion between observed users and new users, refer to the paper for details.
* `AL_Eval_Settings`:
   * `max_selection`: The total number of features that can be selected during testing (Equivalent to EDDI active prediction [EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE](https://arxiv.org/abs/1809.11142))
* `Pretrain_Settings`:
   * `flag_pretrain`: Whether to enable model pretrain
   * `pretrain_number`: The proportion of the entire data set that is used as pre-train data set
* `KL_Schedule_Settings` and `KL_Schedule_Pretrain_Settings`: This is to to anneal the KL penalty term in ELBO. As this schedule is not used in the experiment, they can be ignored. 
