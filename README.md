# Icebreaker
## Overview
This code is based on the paper [Icebreaker: Element-wise Active Information Acquisition with Bayesian Deep Latent Gaussian Model](https://arxiv.org/abs/1908.04537). This repo is divided into two parts:

* PA-BELGAM: Experiments with UCI prediction alone (For users that are only interested in the model rather than active acquisition)
* Icebreaker: Experiments used in the paper (PA-BELGAM + Active acquisition) including UCI imputation, UCI active prediction and Movielens-1m imputation.

## Dependency
This code is implemented with Python 3.6 with the following packages installed:

* [Numpy](https://numpy.org)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [PrettyTable](https://pypi.org/project/PrettyTable/)
* [Pytorch 1.2](https://pytorch.org/get-started/locally/)

## File Structure
`Icebreaker` contains the model and training files for Icebreaker (PA-BELGAM + Active acquisition) experiments:
 
 * `Config`: Contain the **YAML** file that specify the model topology, active learning hyperparameters and some training parameters ([Config Explanation](https://github.com/WenboGong/Icebreaker_Final/tree/master/Icebreaker/Config)):
   * `Config_SGHMC_PNP_UCI.yaml`: Specify configs for the UCI Imputation experiment
   * `Config_SGHMC_PNP_UCI0.yaml`: Specify configs for UCI active prediction experiment
   * `Config_SGHMC_PNP_movielens1m.yaml`: Specify configs for Movielens-1m imputation experiment
* `Dataloader`: Contain the data file and relevant loader and mask functions
   * `data`: where the data file is stored (empty, you shold download the required data)
   * `base_Dataloader.py`: Contain the definition of dataloader
   * `base_mask.py:` Contain the mask functions 
* `Results`: where the results are stored
  * `UCI0`: For UCI imputation and active prediction results
* `base_Model`: Contain the model definition and inference methods
  * `BNN_Network_zoo.py`: Contains the model definitions used in experiments
  * `base_Active_Learning.py`: Define the active learning object for active prediction.
  * `base_Active_Learning_SGHMC.py`: Define the active learning object for training active acquisition (Icebreaker)
  * `base_BNN.py`: Some low-level definitions for layers used in `BNN_Network_zoo.py`
  * `base_Infer.py`: Define inference object for models in `BNN_Network_zoo.py`
  * `base_Network.py`: Some low-level definitions for MLP layers
* `Debug_Retrain_Active_SGHMC_Movielens.py`: Training and testing file for Movielens-1m imputation
* `Debug_Retrain_Active_SGHMC_UCI_Imputation.py`: Training and test file for UCI imputation
* `Debug_Retrain_Active_SGHMC_UCI_Prediction.py`: Training and test file for UCI Active Prediction

`PA-BELGAM` contains the relevant files for PA-BELGAM UCI prediction:

* `Results`: contains the result files
* `PA_BELGAM_Dataloader.py`: Dataloader definitions
* `PA_BELGAM_infer.py`: Inference definition
* `PA_BELGAM_model.py`: Model definition file
* `UCI_Training.py`: Training and test files for UCI prediction

`Util` contains some utility functions for both `Icebreaker` and `PA-BELGAM`.

## Running
### Icebreaker Experiment
#### UCI Imputation
In this example, we use the Boston house dataset. (https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)
Preparing the data: we store the data under Dataloader/data/uci/d0.xls as default and we require the variable names to be put in the first row of the .xls file.
To run the UCI imputation experiment, there are several hyperparameters that need to be set. For example, `python Icebreaker/Debug_Retrain_Active_SGHMC_UCI_Imputation.py --epochs 2000 --uci '0' --uci_normal
    'True_Normal' --seed 130 --step_sghmc 0.0003 --sigma 0.4 --BALD_Coef 1.0 --Conditional_coef
    1.0 --scale_data 1.0 --noisy_update 1.0`.
    
Each hyperparemeter has following meaning:

* `epochs`: The training epochs for the model
* `uci`: the UCI data number: '0' represents Boston Housing
* `uci_normal`: The normalization method for UCI data. 'True_Normal' is to normalize the data with 0 mean and unit variance for each dimension.
* `seed`: The random seed for the experiment
* `step_sghmc`: The discretization step size for SGHMC
* `sigma`: The output sigma for the decoder (shared across all output dimensions).
* `BALD_Coef`: This specifies the coefficient of the acquisition objective in the paper. (No need to change in imputation experiment, set it to 1.) 
*  `Conditional_coef`: The coefficient of the combined ELBO objectives that balances the imputation and prediction (No need to change in imputation experiment, set it to 1. For details, refer to [Learning Structured Output Representation using Deep Conditional Generative Models](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf))
* `scale_data`: Replication of the training data set, set to 1.
* `noisy_update`: The SGHMC burn-in method. '1' represents injecting noise at each step during burn-in. '0': no noise injecting, equivalent to running an optimization method. 
#### UCI Prediction
To run UCI active prediction, `python Debug_Retrain_Active_SGHMC_Opposite.py --epochs 1500 --seed 50 --uci '0'
    --uci_normal 'True_Normal' --step_sghmc 0.0003 --sigma 0.4 --BALD_Coef 0.5 --Conditional_coef
    0.8 --scale_data 1.0 --noisy_update 0.0`
    
The hyperparameter settings is simlar to **UCI imputation** with two main differences:

* `BALD_Coef`
* `Conditional_coef`

#### Movielens-1m Imputation
To run Movielens-1m imputation, it is similar to **UCI imputation**.

**However, if this is your first time to run this file, you should un-comment the following code:**

```
# filename=cwd+'/Dataloader/data/movielens'
# base_Movie=base_movielens(filename,download=True,flag_initial=True,pre_process_method='Fixed',flag_pickle=True)
```

This will automatically download the Movielens-1m dataset and preprocess it into the required format. After this, comment this two lines in later runs as this preprocess is time-consuming. 

### PA-BELGAM Prediction without Active acquisition
This experiment only focuses on the PA-BELGAM part where the active training acquisition is disabled. In other words, this experiment shows the advantages of full Bayesian treatment of the decoder weight compared to point estimate. We also test PA-BELGAM using different size of training data by random sub-sampling.
#### PA-BELGAM UCI prediction

To run the PA-BELGAM alone for UCI prediction, run `python UCI_Training.py`. As there are only a few hyperparameters to be set, you can directly modify them in the training file. 

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.
