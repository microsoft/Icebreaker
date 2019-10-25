'''
This file implements the Active training by retraining the PNP model from scratch using SGHMC
'''

import sys
import os
cwd=os.getcwd()
cwd_parent=os.path.abspath('..')
sys.path.append(cwd)
sys.path.append(cwd_parent)
sys.path.append(cwd+'/base_Model')
sys.path.append(cwd+'/Dataloader')
sys.path.append(cwd_parent+'/Util')

from base_Model.base_Network import *


import argparse

from Icebreaker.Dataloader.base_Dataloader import *
from Icebreaker.Dataloader.base_mask import *
from torch.utils.data import DataLoader

from Icebreaker.base_Model.base_BNN import *
from Icebreaker.base_Model.BNN_Network_zoo import *
import copy
from torch.autograd import grad
from scipy.stats import bernoulli
from Icebreaker.base_Model.base_Active_Learning import *
from Icebreaker.base_Model.base_Active_Learning_SGHMC import *
import random


torch.set_default_tensor_type('torch.cuda.FloatTensor')


parser = argparse.ArgumentParser(description='SEDDI Active Learning + Testing')
parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                    help='number of epochs to train (default: 1500)')
parser.add_argument('--lr_Adam', type=float, default=0.003, metavar='LR_ADAM',
                    help='learning rate for Adam (default: 0.003)')
parser.add_argument('--lr_sghmc', type=float, default=0.003, metavar='LR_SGHMC_ADAM',
                    help='learning rate for sghmc Adam (encoder update) (default: 0.003)')
parser.add_argument('--step_sghmc', type=float, default=4e-4, metavar='LR_SGHMC',
                    help='step size for sghmc (default: 3e-5)')

parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed list (default: 1)')
parser.add_argument('--sigma', type=float, default=0.4, metavar='SIGMA',
                    help='What sigma output should be used default:0.4')

parser.add_argument('--BALD_Coef', type=float, default=0.5, metavar='BALD',
                    help='What is the BALD Coef used Default: 0.5')

parser.add_argument('--Conditional_coef', type=float, default=0.8, metavar='COND',
                    help='The Conditional coef for sghmc and pnp')

parser.add_argument('--uci', type=str, default='0', metavar='UCI',
                    help='UCI number')
parser.add_argument('--uci_normal', type=str, default='True_Normal', metavar='UCI_Normal',
                    help='UCI normalization')

parser.add_argument('--scale_data', type=float, default=1, metavar='SCALE',
                    help='Scale data with default 1')


parser.add_argument('--noisy_update', type=float, default=0., metavar='NOISY',
                    help='whether perform noisy optimization default: clean opt')


args = parser.parse_args()

Select_Split=True

selection_scheme='overall'
flag_LV=False

batch_select=0# Disable this


Strategy='Opposite_Alpha'
# Enable strong baseline to replace RAND
flag_BALD_row=False

if flag_LV:
    raise RuntimeError('flag_LV must be False')
else:
    suffix='%s_%s_'%(selection_scheme,Strategy)

if selection_scheme=='per_user' or selection_scheme=='batch':
    Select_Split=False
# Meta hyperparameter


# Read cfg file
code_path=cwd
print('code_path:%s'%(code_path))
cfg=ReadYAML(cwd+'/Config/Config_SGHMC_PNP_UCI0.yaml')

# Overwrite by parser

rand_seed_list=[args.seed]
num_runs=1

filepath_out = cwd+'/Results/UCI0/Prediction'

# Define the base dataset
# Dataset
uci_number=args.uci
filepath = cwd+'/Dataloader/data/uci/'
base_UCI_obj = base_UCI(uci_number, filepath, test_size=0.25, flag_normalize=True, rs=rand_seed_list[0], scheme='1', min_data=1.,normalization=args.uci_normal)



# Define the model and experiment parameters
# Define model parameters
max_selection_eval=cfg['AL_Eval_Settings']['max_selection'] # This for Active learning of evaluation
encoder_layer_num_before_agg=cfg['BNN_Settings']['encoder_settings']['encoder_layer_num_before_agg']
encoder_hidden_before_agg=cfg['BNN_Settings']['encoder_settings']['encoder_hidden_before_agg']
encoder_layer_num_after_agg=cfg['BNN_Settings']['encoder_settings']['encoder_layer_num_after_agg']
encoder_hidden_after_agg=cfg['BNN_Settings']['encoder_settings']['encoder_hidden_after_agg']
decoder_layer_num=cfg['BNN_Settings']['decoder_settings']['decoder_layer_num']
decoder_hidden=cfg['BNN_Settings']['decoder_settings']['decoder_hidden']
pooling=cfg['BNN_Settings']['encoder_settings']['pooling']
output_const=cfg['BNN_Settings']['decoder_settings']['output_const']
sample_z=cfg['BNN_Settings']['encoder_settings']['sample_z']
sample_W=cfg['BNN_Settings']['decoder_settings']['sample_W']
sample_W_PNP=1
pooling_act=cfg['BNN_Settings']['encoder_settings']['pooling_act']
BNN_init_range=cfg['BNN_Settings']['decoder_settings']['init_range']
BNN_coef_sample=cfg['BNN_Settings']['decoder_settings']['coef_sample']
KL_coef=cfg['BNN_Settings']['KL_coef']
W_sigma_prior=cfg['Training_Settings']['W_sigma_prior']
sigma_out=args.sigma
obs_dim=base_UCI_obj.Data_mat.shape[1]
latent_dim=cfg['BNN_Settings']['latent_dim']
dim_before_agg=cfg['BNN_Settings']['dim_before_agg']
embedding_dim=cfg['BNN_Settings']['embedding_dim']
add_const=cfg['BNN_Settings']['decoder_settings']['add_const']
flag_log_q=cfg['BNN_Settings']['flag_log_q']

flag_clear_target_train=cfg['Active_Learning_Settings']['flag_clear_target_train']
flag_clear_target_test=cfg['Active_Learning_Settings']['flag_clear_target_test']

flag_hybrid=cfg['Active_Learning_Settings']['flag_hybrid']
conditional_coef=args.Conditional_coef
balance_coef=cfg['Active_Learning_Settings']['balance_coef']
BALD_coef=args.BALD_Coef
# KL Schedule
KL_Schedule_Settings=cfg['KL_Schedule_Settings']
KL_coef_W=None#KL_Schedule_helper(**KL_Schedule_Settings)
# KL Pretrain_Schedule
KL_Schedule_Pretrain_Settings=cfg['KL_Schedule_Pretrain_Settings']
KL_coef_W_pretrain=None#KL_Schedule_helper(**KL_Schedule_Pretrain_Settings)

max_selection=cfg['Active_Learning_Settings']['max_selection'] # This for active learning of W
flag_pretrain=cfg['Pretrain_Settings']['flag_pretrain']
step = cfg['Active_Learning_Settings']['step']

# Store the Results
RMSE_BALD_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
MAE_BALD_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
NLL_BALD_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))

RMSE_BALD_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
MAE_BALD_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
NLL_BALD_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))

RMSE_RAND_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
MAE_RAND_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
NLL_RAND_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))

RMSE_RAND_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
MAE_RAND_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
NLL_RAND_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))

RMSE_RAND_PNP_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
MAE_RAND_PNP_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
NLL_RAND_PNP_mat_BALD=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))

RMSE_RAND_PNP_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
MAE_RAND_PNP_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))
NLL_RAND_PNP_mat_RAND=np.zeros((num_runs,math.ceil(max_selection/step)+1,max_selection_eval+1))

print('max_Selection:%s'%(max_selection_eval))

cfg['Optimizer']['lr_sghmc']=args.lr_sghmc
cfg['Optimizer']['lr']=args.lr_Adam

for runs in range(num_runs):
    counter_selection=0

    # Set random seed
    rand_seed = rand_seed_list[runs]
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    torch.backends.cudnn.deterministic = True

    ########### ########## ########## ########## ########## ########## ########## Pretrain the model ########## ########## ########## ########## ########## ########## ##########
    # Define dataset
    base_UCI_obj = base_UCI(uci_number, filepath, test_size=0.25, flag_normalize=True, rs=rand_seed, scheme='1',
                                min_data=1.,normalization=args.uci_normal)
    # Setup table
    table = PrettyTable()
    table.field_names = ['Data', 'AUIC', 'AUIC_Diff', 'AUIC_Diff_R', 'AUIC_Diff_P','SP BALD','SP RAND','SP RANDP', 'Observed_Diff', 'Observed BALD',
                         'Initial Test Choice BALD', 'Initial Test Choice RAND', 'Initial Test Choice RANDP','Out BALD','Out RAND','Out RANDP']

    counter_loop=0

    # Define Model
    PNP_SGHMC = Point_Net_Plus_BNN_SGHMC(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                         encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                         encoder_hidden_before_agg=encoder_hidden_before_agg, \
                                         encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                         encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                         decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden, \
                                         pooling=pooling, output_const=output_const, add_const=add_const,
                                         sample_z=sample_z, sample_W=sample_W, W_sigma_prior=W_sigma_prior,
                                         pooling_act=pooling_act, flag_log_q=flag_log_q)
    PNP_SGHMC_RAND = Point_Net_Plus_BNN_SGHMC(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                         encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                         encoder_hidden_before_agg=encoder_hidden_before_agg, \
                                         encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                         encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                         decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden, \
                                         pooling=pooling, output_const=output_const, add_const=add_const,
                                         sample_z=sample_z, sample_W=sample_W, W_sigma_prior=W_sigma_prior,
                                         pooling_act=pooling_act, flag_log_q=flag_log_q)

    PNP_Det = Point_Net_Plus_BNN(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                 encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                 encoder_hidden_before_agg=encoder_hidden_before_agg,
                                 encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                 encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                 decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden,
                                 pooling=pooling, output_const=output_const, add_const=add_const, sample_z=sample_z,
                                 sample_W=sample_W_PNP, W_sigma_prior=W_sigma_prior, pooling_act=pooling_act,
                                 BNN_init_range=BNN_init_range, BNN_coef_sample=0.,
                                 KL_coef=KL_coef, flag_local=True, flag_log_q=flag_log_q
                                 )
    #Infer Model


    Infer_SGHMC = SGHMC(model=PNP_SGHMC, Infer_name='Scale Adapted SGHMC')
    Infer_SGHMC_RAND = SGHMC(model=PNP_SGHMC_RAND, Infer_name='Scale Adapted SGHMC RAND')
    # Parameter List
    list_p_z = list(PNP_SGHMC.encoder_before_agg.parameters()) + list(PNP_SGHMC.encoder_after_agg.parameters()) + [
        PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias]
    list_p_z_RAND = list(PNP_SGHMC_RAND.encoder_before_agg.parameters()) + list(PNP_SGHMC_RAND.encoder_after_agg.parameters()) + [
        PNP_SGHMC_RAND.encode_embedding, PNP_SGHMC_RAND.encode_bias]


    Adam_encoder = torch.optim.Adam(
        list(PNP_SGHMC.encoder_before_agg.parameters()) + list(PNP_SGHMC.encoder_after_agg.parameters()), lr=cfg['Optimizer']['lr_sghmc'],
        betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
        weight_decay=cfg['Optimizer']['weight_decay'])
    Adam_embedding = torch.optim.Adam([PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias], lr=cfg['Optimizer']['lr_sghmc'],
                                      betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                      weight_decay=cfg['Optimizer']['weight_decay'])
    # BNN_RAND
    Adam_encoder_RAND = torch.optim.Adam(
        list(PNP_SGHMC_RAND.encoder_before_agg.parameters()) + list(PNP_SGHMC_RAND.encoder_after_agg.parameters()),
        lr=cfg['Optimizer']['lr_sghmc'],
        betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
        weight_decay=cfg['Optimizer']['weight_decay'])

    Adam_embedding_RAND = torch.optim.Adam([PNP_SGHMC_RAND.encode_embedding, PNP_SGHMC_RAND.encode_bias], lr=cfg['Optimizer']['lr_sghmc'],
                                           betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                           weight_decay=cfg['Optimizer']['weight_decay'])
    # PNP_RAND
    Adam_encoder_RAND_PNP = torch.optim.Adam(
        list(PNP_Det.encoder_before_agg.parameters()) + list(PNP_Det.encoder_after_agg.parameters()),
        lr=cfg['Optimizer']['lr'],
        betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
        weight_decay=cfg['Optimizer']['weight_decay'])
    Adam_decoder_RAND_PNP = torch.optim.Adam(list(PNP_Det.decoder.parameters()), lr=cfg['Optimizer']['lr'],
                                             betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                             weight_decay=cfg['Optimizer']['weight_decay'])
    Adam_embedding_RAND_PNP = torch.optim.Adam([PNP_Det.encode_embedding, PNP_Det.encode_bias],
                                               lr=cfg['Optimizer']['lr'],
                                               betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                               weight_decay=cfg['Optimizer']['weight_decay'])

    # Load Active Learning Settings
    # How many data points to select for each time
    num_selected_variable = cfg['Active_Learning_Settings']['step']
    sample_x = cfg['Active_Learning_Settings']['sample_x']
    # Load other settings
    Dict_training_settings = cfg['Training_Settings']
    Dict_training_settings_PNP = copy.deepcopy(Dict_training_settings)
    Dict_training_settings_PNP['flag_BNN'] = False  # This is to turn off the BNN training for PNP
    Dict_training_settings_PNP['flag_stream'] = False

    Dict_dataset_settings = cfg['Dataset_Settings']

    # DEFINE Active Learning obj
    Active_BALD = base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC,Infer_model=Infer_SGHMC,overall_data=base_UCI_obj.Data_mat,rs=rand_seed,sigma_out=sigma_out,Optim_settings=cfg['Optimizer'],Adam_encoder=Adam_encoder,
                                                 Adam_embedding=Adam_embedding,flag_clear_target_train=flag_clear_target_train,flag_clear_target_test=flag_clear_target_test,model_name='SGHMC Active')
    Active_RAND=base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC_RAND,Infer_model=Infer_SGHMC_RAND,overall_data=base_UCI_obj.Data_mat,rs=rand_seed,sigma_out=sigma_out,Optim_settings=cfg['Optimizer'],Adam_encoder=Adam_encoder_RAND,
                                                 Adam_embedding=Adam_embedding_RAND,flag_clear_target_train=flag_clear_target_train,flag_clear_target_test=flag_clear_target_test,model_name='SGHMC Active RAND')

    Active_RAND_PNP = base_Active_Learning_Decoder(model=PNP_Det, overall_data=base_UCI_obj.Data_mat, active_scheme='1',
                                                   rs=rand_seed, Adam_encoder=Adam_encoder_RAND_PNP,
                                                   Adam_decoder=Adam_decoder_RAND_PNP,
                                                   Adam_embedding=Adam_embedding_RAND_PNP,
                                                   flag_clear_target_train=flag_clear_target_train,
                                                   flag_clear_target_test=flag_clear_target_test,
                                                   sigma_out=sigma_out,
                                                   Optim_settings=cfg['Optimizer'], model_name='RAND_PNP'
                                                   )

    # Pre-Process the data (Only Define Once)
    Active_BALD._data_preprocess(**Dict_dataset_settings)

    # filepathname = '/mnt/_default/dataset/uci/Train_Data_UCI%s/Seed_20_' % (args.uci)
    # Active_BALD._load_data(filepathname=filepathname)

    Active_RAND.test_input_tensor = torch.tensor(Active_BALD.test_input_tensor.data)
    Active_RAND.test_target_tensor = torch.tensor(Active_BALD.test_target_tensor.data)
    Active_RAND_PNP.test_input_tensor = torch.tensor(Active_BALD.test_input_tensor.data)
    Active_RAND_PNP.test_target_tensor = torch.tensor(Active_BALD.test_target_tensor.data)

    # Get pretrain data
    Active_BALD._get_pretrain_data(pretrain_number=cfg['Pretrain_Settings']['pretrain_number'])
    pretrain_data = Active_BALD.pretrain_data_tensor.clone().detach()

    valid_data_tensor=Active_BALD.valid_data_tensor.clone().detach()
    valid_data_input_tensor=None#Active_BALD.valid_data_input_tensor.clone().detach()
    valid_data_target_tensor=Active_BALD.valid_data_target_tensor.clone().detach()
    train_full_pool = torch.tensor(Active_BALD.train_data_tensor.data)
    flag_reset_optim=Dict_training_settings['flag_reset_optim']
    # Overwrite the flag_stream
    Dict_training_settings_PNP['flag_stream'] = False
    Drop_orig = Dict_training_settings['Drop_p']
    Drop_p=0.6
    Dict_training_settings_PNP['Drop_p'] = Drop_p

    epoch_orig = args.epochs
    tot_epoch=args.epochs
    Dict_training_settings_PNP['epoch'] = tot_epoch+500
    Dict_training_settings_PNP['sigma_out']=sigma_out

    conditional_coef=args.Conditional_coef
    conditional_coef_sghmc=args.Conditional_coef



    # Load the stored data for debug purpose
    # filename_data = cfg['File_Settings']['Result_path']
    # #pretrain_data=np.load(filename_data+'Stored_Data/Active_Learning_W_UCI1_SGHMC_ep2000_step50_total1500_CTTest_Conditional0.8_BALD0.92_Balance0.5_Temp0.25p0_Drop0.2_Sigma0.3_UpdateEncoder_WeightTar1_eps3e-5_Adam0.003_ProbSelect_ALTest_BALD_datanum_1000_Seed_730.npy')
    # pretrain_data=np.load(filename_data+'Stored_Data/Active_Learning_W_UCI1_SGHMC_ep2000_step25_total1000_CTTest_Conditional0.8_BALD0.92_Balance0.5_Temp0.25p0_Drop0.2_Sigma0.4_UpdateEncoder_WeightTar1_eps5e-5_Adam0.003_ProbSelect_Alpha_ALTest_BALD_datanum_1000_Seed_730.npy')
    # pretrain_data=torch.from_numpy(pretrain_data).float().cuda()
    #
    # # Pretrain
    Active_RAND_PNP.train_BNN( flag_pretrain=True,
                              observed_train=pretrain_data,
                              random_seed=rand_seed, KL_coef_W=KL_coef_W_pretrain, flag_hybrid=flag_hybrid,
                              target_dim=-1,valid_data=valid_data_input_tensor,valid_data_target=valid_data_target_tensor,conditional_coef=conditional_coef,
                              **Dict_training_settings_PNP)





    W_sample_RAND = Active_RAND.train_BNN(pretrain_data, eps=args.step_sghmc, max_sample_size=40, tot_epoch=tot_epoch+500,
                                          thinning=10,
                                          hyper_param_update=25000, sample_int=10,
                                          flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder_RAND,
                                          Adam_embedding=Adam_embedding_RAND, Drop_p=Drop_p,
                                          list_p_z=list_p_z_RAND,
                                          test_input_tensor=Active_RAND.test_input_tensor,
                                          test_target_tensor=Active_RAND.test_target_tensor, conditional_coef=conditional_coef_sghmc,
                                          target_dim=-1,flag_reset_optim=flag_reset_optim,valid_data=valid_data_input_tensor,valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update)

    W_sample = Active_BALD.train_BNN(pretrain_data, eps=args.step_sghmc, max_sample_size=40, tot_epoch=tot_epoch+500,
                                     thinning=10,
                                     hyper_param_update=25000, sample_int=10,
                                     flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder,
                                     Adam_embedding=Adam_embedding, Drop_p=Drop_p, list_p_z=list_p_z,
                                     test_input_tensor=Active_BALD.test_input_tensor,
                                     test_target_tensor=Active_BALD.test_target_tensor,
                                     conditional_coef=conditional_coef_sghmc,
                                     target_dim=-1, flag_reset_optim=flag_reset_optim,
                                     valid_data=valid_data_input_tensor,
                                     valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update)

    # Write back
    Dict_training_settings_PNP['Drop_p'] = Drop_orig
    Dict_training_settings_PNP['epoch'] = epoch_orig+500
    # Initialized samples
    W_dict_init=None
    W_dict_init_RAND=None

    ########## ########## ########## ########## ########## ########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

    #################################################### Prepare the training data  ##############################################################################
    # BALD
    train_pool_data_BALD = Active_BALD.train_pool_tensor.clone().detach()
    train_full_pool = Active_BALD.train_data_tensor.clone().detach()
    observed_train_BALD = Active_BALD.init_observed_train_tensor.clone().detach()
    # RAND
    train_pool_data_RAND = torch.tensor(Active_BALD.train_pool_tensor.data)
    observed_train_RAND = torch.tensor(Active_BALD.init_observed_train_tensor.data)
    # PNP RAND
    train_pool_data_RAND_PNP = torch.tensor(Active_BALD.train_pool_tensor.data)
    observed_train_RAND_PNP = torch.tensor(Active_BALD.init_observed_train_tensor.data)

    train_data_perm=Active_BALD.train_data_tensor.clone().detach()
    # Initialization for test data
    test_pool_perm=Active_BALD.test_input_tensor.clone().detach()
    test_input_perm=torch.zeros(test_pool_perm.shape)
    test_pool_data_BALD_BALD = torch.tensor(Active_BALD.test_input_tensor.data)
    test_pool_data_BALD_RAND = torch.tensor(Active_BALD.test_input_tensor.data)
    test_pool_data_RAND_BALD = torch.tensor(Active_BALD.test_input_tensor.data)
    test_pool_data_RAND_RAND = torch.tensor(Active_BALD.test_input_tensor.data)
    test_pool_data_RAND_PNP_BALD = torch.tensor(Active_BALD.test_input_tensor.data)
    test_pool_data_RAND_PNP_RAND = torch.tensor(Active_BALD.test_input_tensor.data)

    test_target_data = Active_BALD.test_target_tensor.clone().detach()

    # Initialize test input
    test_input_BALD_BALD = torch.zeros(Active_BALD.test_input_tensor.shape)  # N_test x obs_dim
    test_input_BALD_RAND = torch.zeros(Active_BALD.test_input_tensor.shape)  # N_test x obs_dim
    test_input_RAND_BALD = torch.zeros(Active_BALD.test_input_tensor.shape)
    test_input_RAND_RAND = torch.zeros(Active_BALD.test_input_tensor.shape)
    test_input_RAND_PNP_BALD = torch.zeros(Active_BALD.test_input_tensor.shape)
    test_input_RAND_PNP_RAND = torch.zeros(Active_BALD.test_input_tensor.shape)




    ############################################################################################################################################################


    ################################################### Evaluation of the pretrain model ########################################################################
    # BALD_BALD
    RMSE_BALD_BALD, MAE_BALD_BALD, NLL_BALD_BALD = test_UCI_AL(model=Active_BALD, max_selection=max_selection_eval,
                                                                    sample_x=None,
                                                                    test_input=test_input_BALD_BALD,
                                                                    test_pool=test_pool_data_BALD_BALD,
                                                                    test_target=test_target_data, sigma_out=sigma_out,
                                                                    search='Target',
                                                                    model_name='PNP_SGHMC', W_sample=W_sample)
    # BALD_RAND
    RMSE_BALD_RAND, MAE_BALD_RAND, NLL_BALD_RAND = test_UCI_AL(model=Active_BALD, max_selection=max_selection_eval,
                                                                    sample_x=None,
                                                                    test_input=test_input_BALD_RAND,
                                                                    test_pool=test_pool_data_BALD_RAND,
                                                                    test_target=test_target_data, sigma_out=sigma_out,
                                                                    search='Random',
                                                                    model_name='PNP_SGHMC', W_sample=W_sample)
    # RAND_BALD
    RMSE_RAND_BALD, MAE_RAND_BALD, NLL_RAND_BALD = test_UCI_AL(model=Active_RAND, max_selection=max_selection_eval,
                                                                    sample_x=None,
                                                                    test_input=test_input_RAND_BALD,
                                                                    test_pool=test_pool_data_RAND_BALD,
                                                                    test_target=test_target_data, sigma_out=sigma_out,
                                                                    search='Target',
                                                                    model_name='PNP_SGHMC', W_sample=W_sample_RAND)

    # RAND_RAND
    RMSE_RAND_RAND, MAE_RAND_RAND, NLL_RAND_RAND = test_UCI_AL(model=Active_RAND, max_selection=max_selection_eval,
                                                                    sample_x=None,
                                                                    test_input=test_input_RAND_RAND,
                                                                    test_pool=test_pool_data_RAND_RAND,
                                                                    test_target=test_target_data, sigma_out=sigma_out,
                                                                    search='Random',
                                                                    model_name='PNP_SGHMC', W_sample=W_sample_RAND)
    # RAND_PNP_BALD
    RMSE_RAND_PNP_BALD, MAE_RAND_PNP_BALD, NLL_RAND_PNP_BALD = test_UCI_AL(model=Active_RAND_PNP, max_selection=max_selection_eval,
                                                                    sample_x=200,
                                                                    test_input=test_input_RAND_PNP_BALD,
                                                                    test_pool=test_pool_data_RAND_PNP_BALD,
                                                                    test_target=test_target_data, sigma_out=sigma_out,
                                                                    search='Target',
                                                                    model_name='PNP_BNN', W_sample=None)
    # PNP_RAND_RAND
    RMSE_RAND_PNP_RAND, MAE_RAND_PNP_RAND, NLL_RAND_PNP_RAND = test_UCI_AL(model=Active_RAND_PNP, max_selection=max_selection_eval,
                                                                    sample_x=200,
                                                                    test_input=test_input_RAND_PNP_RAND,
                                                                    test_pool=test_pool_data_RAND_PNP_RAND,
                                                                    test_target=test_target_data, sigma_out=sigma_out,
                                                                    search='Random',
                                                                    model_name='PNP_BNN', W_sample=None)

    # Compute AUIC
    AUIC_Diff_BALD = Compute_AUIC_1D(const=None, BALD=NLL_BALD_BALD, RAND=NLL_BALD_RAND)
    AUIC_Diff_RAND = Compute_AUIC_1D(const=None, BALD=NLL_RAND_BALD, RAND=NLL_RAND_RAND)
    AUIC_Diff_RAND_PNP = Compute_AUIC_1D(const=None, BALD=NLL_RAND_PNP_BALD, RAND=NLL_RAND_PNP_RAND)
    AUIC_BALD_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_BALD_BALD)
    AUIC_RAND_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_RAND_BALD)
    AUIC_RAND_PNP_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_RAND_PNP_BALD)
    AUIC_Diff = Compute_AUIC_Differece(AUIC_BALD_BALD, target1=AUIC_RAND_BALD, target2=AUIC_RAND_PNP_BALD)



    Results_list = ['%s Init' % (0 * step),
                    '%.3f , %.3f , %.3f' % (AUIC_BALD_BALD, AUIC_RAND_BALD, AUIC_RAND_PNP_BALD),
                    '%.3f , %.3f , %.3f' % (AUIC_Diff_BALD, AUIC_Diff_RAND, AUIC_Diff_RAND_PNP)] + AUIC_Diff +\
        ['Full','Full','Full']+\
                   [get_choice(observed_train_BALD)] \
                   + [get_choice(observed_train_BALD)] + [get_choice(test_input_BALD_BALD).tolist()] + [
                       get_choice(test_input_RAND_BALD).tolist()] + [
                       get_choice(test_input_RAND_PNP_BALD).tolist()]+['None, None','None None','None None']
    table = Table_format(table, Results_list)
    # Store the table
    if Select_Split==True:
        setting_num = 'Debug_Step%s_total%s_%s_CTTest_Conditional%s_%sBALD_%sBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_WeightTar1_eps%s_Adam%s_ALTest_%sPTable_'\
    %(step,max_selection,args.uci_normal,conditional_coef,BALD_coef,balance_coef,sigma_out,Drop_orig,args.step_sghmc,args.lr_sghmc,suffix)
    else:
        print('Indeed No Balance')
        setting_num = 'Debug_Step%s_total%s_%s_CTTest_Conditional%s_%sBALD_NoBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_WeightTar1_eps%s_Adam%s_ALTest_%sPTable_' \
                      % (step, max_selection, args.uci_normal, conditional_coef, BALD_coef, sigma_out,
                         Drop_orig, args.step_sghmc, args.lr_sghmc,suffix)

    print('Run:pretrain Data: %s' % ( counter_loop * step))
    print('NLL_BALD_BALD:%s' % (NLL_BALD_BALD))
    print('NLL_BALD_RAND:%s' % (NLL_BALD_RAND))
    print('NLL_PNP_RAND_BALD:%s' % (NLL_RAND_PNP_BALD))
    print('NLL_PNP_RAND_RAND:%s' % (NLL_RAND_PNP_RAND))
    print('NLL_RAND_BALD:%s' % (NLL_RAND_BALD))
    print('NLL_RAND_RAND:%s' % (NLL_RAND_RAND))
    print(
        'AUIC_Diff_BALD: %.3f RAND: %.3f RAND_PNP: %.3f' % (AUIC_Diff_BALD, AUIC_Diff_RAND, AUIC_Diff_RAND_PNP))
    print('AUIC BALD: %.3f RAND: %.3f RAND_PNP: %.3f' % (AUIC_BALD_BALD, AUIC_RAND_BALD, AUIC_RAND_PNP_BALD))

    # Store Results
    store_Table(filepath_out + '/Setting%s_Seed_%s' % (setting_num, rand_seed) + '.txt',
                table, title='UCI_%s_Setting_%s_Seed_%s' % (uci_number, setting_num, rand_seed))

    RMSE_BALD_mat_BALD[runs, counter_loop, :], MAE_BALD_mat_BALD[runs, counter_loop, :], NLL_BALD_mat_BALD[runs,
                                                                                         counter_loop,
                                                                                         :] = RMSE_BALD_BALD, MAE_BALD_BALD, NLL_BALD_BALD
    RMSE_BALD_mat_RAND[runs, counter_loop, :], MAE_BALD_mat_RAND[runs, counter_loop, :], NLL_BALD_mat_RAND[runs,
                                                                                         counter_loop,
                                                                                         :] = RMSE_BALD_RAND, MAE_BALD_RAND, NLL_BALD_RAND
    RMSE_RAND_mat_BALD[runs, counter_loop, :], MAE_RAND_mat_BALD[runs, counter_loop, :], NLL_RAND_mat_BALD[runs,
                                                                                         counter_loop,
                                                                                         :] = RMSE_RAND_BALD, MAE_RAND_BALD, NLL_RAND_BALD
    RMSE_RAND_mat_RAND[runs, counter_loop, :], MAE_RAND_mat_RAND[runs, counter_loop, :], NLL_RAND_mat_RAND[runs,
                                                                                         counter_loop,
                                                                                         :] = RMSE_RAND_RAND, MAE_RAND_RAND, NLL_RAND_RAND
    RMSE_RAND_PNP_mat_BALD[runs, counter_loop, :], MAE_RAND_PNP_mat_BALD[runs, counter_loop,
                                                   :], NLL_RAND_PNP_mat_BALD[runs, counter_loop,
                                                       :] = RMSE_RAND_PNP_BALD, MAE_RAND_PNP_BALD, NLL_RAND_PNP_BALD
    RMSE_RAND_PNP_mat_RAND[runs, counter_loop, :], MAE_RAND_PNP_mat_RAND[runs, counter_loop,
                                                   :], NLL_RAND_PNP_mat_RAND[runs, counter_loop,
                                                       :] = RMSE_RAND_PNP_RAND, MAE_RAND_PNP_RAND, NLL_RAND_PNP_RAND

    counter_loop += 1
    ##################################################################################################################################
    if selection_scheme=='batch' or selection_scheme=='per_user':
        max_selection=torch.sum(torch.abs(Active_BALD.train_pool_tensor)>0.)
        idx_start=0
        idx_end=idx_start+batch_select
    else:
        idx_start=0
        idx_end=0
    ########################################## Doing active selection and retraining ###################################################

    while counter_selection<max_selection:
        if counter_loop == 1:
            flag_init_train = True
        else:
            flag_init_train = False

        # Initialization for test data
        test_pool_data_BALD_BALD = test_pool_perm.clone().detach()
        test_pool_data_BALD_RAND = test_pool_perm.clone().detach()
        test_pool_data_RAND_BALD = test_pool_perm.clone().detach()
        test_pool_data_RAND_RAND = test_pool_perm.clone().detach()
        test_pool_data_RAND_PNP_BALD = test_pool_perm.clone().detach()
        test_pool_data_RAND_PNP_RAND = test_pool_perm.clone().detach()

        test_target_data = test_target_data.clone().detach()
        # Initialize test input
        test_input_BALD_BALD = test_input_perm.clone().detach()  # N_test x obs_dim
        test_input_BALD_RAND = test_input_perm.clone().detach()  # N_test x obs_dim
        test_input_RAND_BALD = test_input_perm.clone().detach()
        test_input_RAND_RAND = test_input_perm.clone().detach()
        test_input_RAND_PNP_BALD = test_input_perm.clone().detach()
        test_input_RAND_PNP_RAND = test_input_perm.clone().detach()
        # Record old train data
        observed_train_BALD_old = observed_train_BALD.clone().detach()
        observed_train_RAND_old = observed_train_RAND.clone().detach()
        observed_train_RAND_PNP_old = observed_train_RAND_PNP.clone().detach()


        # Active selection BALD
        if counter_loop <= 0:
            observed_train_BALD, train_pool_data_BALD, flag_full, num_selected = Active_BALD.base_random_select_training(
                observed_train=observed_train_BALD, pool_data=train_pool_data_BALD, step=num_selected_variable,
             flag_initial=flag_init_train, balance_prop=balance_coef,Select_Split=Select_Split,selection_scheme=selection_scheme)

        else:
            # coef_explor=1./(1+math.exp(0.1*(counter_loop-10)))
            # coef_explor=np.minimum(0.7,np.maximum(coef_explor,0.4))
            # coef_explor=0.3
            # print('Coef:%s'%(coef_explor))
            flag_weight=int((counter_loop+1)%2)
            observed_train_BALD, train_pool_data_BALD, flag_full, num_selected = Active_BALD.base_active_learning_decoder(balance_prop=balance_coef, coef=BALD_coef,
                observed_train=observed_train_BALD, pool_data=train_pool_data_BALD, step=num_selected_variable,
                 flag_initial=flag_init_train, sigma_out=sigma_out, W_sample=W_sample,strategy=Strategy,strategy_alternating=flag_weight,Select_Split=Select_Split,selection_scheme=selection_scheme,\
            idx_start=idx_start,idx_end=idx_end)



        observed_train_BALD = observed_train_BALD.clone().detach()
        train_pool_data_BALD = train_pool_data_BALD.clone().detach()
        # Rand Selection
        if not flag_BALD_row:
            observed_train_RAND, train_pool_data_RAND, flag_full, num_selected = Active_RAND.base_random_select_training(
            observed_train=observed_train_RAND, pool_data=train_pool_data_RAND, step=num_selected_variable,
            balance_prop=balance_coef, flag_initial=flag_init_train,Select_Split=Select_Split,selection_scheme=selection_scheme)
        else:
            observed_train_RAND,train_pool_data_RAND,flag_full =Active_RAND.base_BALD_select_row(step=num_selected_variable,W_sample=W_sample_RAND,pool_data=train_pool_data_RAND,
                                                                                                 observed_train=observed_train_RAND)


        observed_train_RAND = observed_train_RAND.clone().detach()
        train_pool_data_RAND = train_pool_data_RAND.clone().detach()
        # Rand PNP Selection
        observed_train_RAND_PNP, train_pool_data_RAND_PNP, flag_full, num_selected = Active_RAND_PNP.base_random_select_training(
            observed_train=observed_train_RAND_PNP, pool_data=train_pool_data_RAND_PNP, step=num_selected_variable,
            balance_prop=balance_coef, flag_initial=flag_init_train,Select_Split=Select_Split,selection_scheme=selection_scheme
        )

        observed_train_RAND_PNP = observed_train_RAND_PNP.clone().detach()
        train_pool_data_RAND_PNP = train_pool_data_RAND_PNP.clone().detach()
        # If hybrid, apply the target variable
        if flag_hybrid:  # TODO: If target variable is applied
            observed_train_BALD = Active_BALD.get_target_variable(observed_train_BALD, observed_train_BALD_old,
                                                                  target_dim=-1,train_data=train_data_perm)
            observed_train_RAND = Active_BALD.get_target_variable(observed_train_RAND, observed_train_RAND_old,
                                                                  target_dim=-1,train_data=train_data_perm)
            observed_train_RAND_PNP = Active_BALD.get_target_variable(observed_train_RAND_PNP,
                                                                      observed_train_RAND_PNP_old, target_dim=-1,train_data=train_data_perm
                                                                      )
        # Update the selected points
        counter_selection+=num_selected
        # Remove the zeros (rz) in the training data
        observed_train_BALD_rz = remove_zero_row_2D(observed_train_BALD)
        observed_train_RAND_rz = remove_zero_row_2D(observed_train_RAND)
        observed_train_RAND_PNP_rz = remove_zero_row_2D(observed_train_RAND_PNP)

        # Now detect the sp and outlier
        # Now detect the pattern
        BALD_pattern_old, BALD_pattern_new = selection_pattern(observed_train_BALD, observed_train_BALD_old)
        RAND_pattern_old, RAND_pattern_new = selection_pattern(observed_train_RAND, observed_train_RAND_old)
        RAND_PNP_pattern_old, RAND_PNP_pattern_new = selection_pattern(observed_train_RAND_PNP, observed_train_RAND_PNP_old)

        # Now detect the outlier
        BALD_outlier_dim, BALD_outlier_value = outlier_detection(observed_train_BALD, observed_train_BALD_old,
                                                                 diff_scale=1.5)
        RAND_outlier_dim, RAND_outlier_value = outlier_detection(observed_train_RAND, observed_train_RAND_old,
                                                                 diff_scale=1.5)
        RAND_PNP_outlier_dim, RAND_PNP_outlier_value = outlier_detection(observed_train_RAND_PNP,
                                                                         observed_train_RAND_PNP_old, diff_scale=1.5)

        # Now Retrain the model

        # redefine the model
        PNP_SGHMC = Point_Net_Plus_BNN_SGHMC(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                             encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                             encoder_hidden_before_agg=encoder_hidden_before_agg, \
                                             encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                             encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                             decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden, \
                                             pooling=pooling, output_const=output_const, add_const=add_const,
                                             sample_z=sample_z, sample_W=sample_W, W_sigma_prior=W_sigma_prior,
                                             pooling_act=pooling_act, flag_log_q=flag_log_q)
        PNP_SGHMC_RAND = Point_Net_Plus_BNN_SGHMC(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                                  encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                                  encoder_hidden_before_agg=encoder_hidden_before_agg, \
                                                  encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                                  encoder_hidden_after_agg=encoder_hidden_after_agg,
                                                  embedding_dim=embedding_dim,
                                                  decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden, \
                                                  pooling=pooling, output_const=output_const, add_const=add_const,
                                                  sample_z=sample_z, sample_W=sample_W, W_sigma_prior=W_sigma_prior,
                                                  pooling_act=pooling_act, flag_log_q=flag_log_q)

        PNP_Det = Point_Net_Plus_BNN(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                     encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                     encoder_hidden_before_agg=encoder_hidden_before_agg,
                                     encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                     encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                     decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden,
                                     pooling=pooling, output_const=output_const, add_const=add_const, sample_z=sample_z,
                                     sample_W=sample_W_PNP, W_sigma_prior=W_sigma_prior, pooling_act=pooling_act,
                                     BNN_init_range=BNN_init_range, BNN_coef_sample=0.,
                                     KL_coef=KL_coef, flag_local=True, flag_log_q=flag_log_q
                                     )
        # Infer Model
        Infer_SGHMC = SGHMC(model=PNP_SGHMC, Infer_name='Scale Adapted SGHMC')
        Infer_SGHMC_RAND = SGHMC(model=PNP_SGHMC_RAND, Infer_name='Scale Adapted SGHMC RAND')
        # Parameter List
        list_p_z = list(PNP_SGHMC.encoder_before_agg.parameters()) + list(PNP_SGHMC.encoder_after_agg.parameters()) + [
            PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias]
        list_p_z_RAND = list(PNP_SGHMC_RAND.encoder_before_agg.parameters()) + list(
            PNP_SGHMC_RAND.encoder_after_agg.parameters()) + [
                            PNP_SGHMC_RAND.encode_embedding, PNP_SGHMC_RAND.encode_bias]

        Adam_encoder = torch.optim.Adam(
            list(PNP_SGHMC.encoder_before_agg.parameters()) + list(PNP_SGHMC.encoder_after_agg.parameters()),
            lr=cfg['Optimizer']['lr_sghmc'],
            betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
            weight_decay=cfg['Optimizer']['weight_decay'])
        Adam_embedding = torch.optim.Adam([PNP_SGHMC.encode_embedding, PNP_SGHMC.encode_bias], lr=cfg['Optimizer']['lr_sghmc'],
                                          betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                          weight_decay=cfg['Optimizer']['weight_decay'])
        # BNN_RAND
        Adam_encoder_RAND = torch.optim.Adam(
            list(PNP_SGHMC_RAND.encoder_before_agg.parameters()) + list(PNP_SGHMC_RAND.encoder_after_agg.parameters()),
            lr=cfg['Optimizer']['lr_sghmc'],
            betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
            weight_decay=cfg['Optimizer']['weight_decay'])

        Adam_embedding_RAND = torch.optim.Adam([PNP_SGHMC_RAND.encode_embedding, PNP_SGHMC_RAND.encode_bias],
                                               lr=cfg['Optimizer']['lr_sghmc'],
                                               betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                               weight_decay=cfg['Optimizer']['weight_decay'])
        # PNP_RAND
        Adam_encoder_RAND_PNP = torch.optim.Adam(
            list(PNP_Det.encoder_before_agg.parameters()) + list(PNP_Det.encoder_after_agg.parameters()),
            lr=cfg['Optimizer']['lr'],
            betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
            weight_decay=cfg['Optimizer']['weight_decay'])
        Adam_decoder_RAND_PNP = torch.optim.Adam(list(PNP_Det.decoder.parameters()), lr=cfg['Optimizer']['lr'],
                                                 betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                                 weight_decay=cfg['Optimizer']['weight_decay'])
        Adam_embedding_RAND_PNP = torch.optim.Adam([PNP_Det.encode_embedding, PNP_Det.encode_bias],
                                                   lr=cfg['Optimizer']['lr'],
                                                   betas=(cfg['Optimizer']['beta1'], cfg['Optimizer']['beta2']),
                                                   weight_decay=cfg['Optimizer']['weight_decay'])
        # Define Active obj
        Active_BALD = base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC, Infer_model=Infer_SGHMC,
                                                         overall_data=base_UCI_obj.Data_mat, rs=rand_seed, sigma_out=sigma_out,
                                                         Optim_settings=cfg['Optimizer'], Adam_encoder=Adam_encoder,
                                                         Adam_embedding=Adam_embedding, flag_clear_target_train=flag_clear_target_train,flag_clear_target_test=flag_clear_target_test,
                                                         model_name='SGHMC Active')
        Active_RAND = base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC_RAND, Infer_model=Infer_SGHMC_RAND,
                                                         overall_data=base_UCI_obj.Data_mat, rs=rand_seed, sigma_out=sigma_out,
                                                         Optim_settings=cfg['Optimizer'], Adam_encoder=Adam_encoder_RAND,
                                                         Adam_embedding=Adam_embedding_RAND,
                                                         flag_clear_target_train=flag_clear_target_train,
                                                         flag_clear_target_test=flag_clear_target_test,
                                                         model_name='SGHMC Active RAND')

        Active_RAND_PNP = base_Active_Learning_Decoder(model=PNP_Det, overall_data=base_UCI_obj.Data_mat, active_scheme='1',
                                                       rs=rand_seed, Adam_encoder=Adam_encoder_RAND_PNP,
                                                       Adam_decoder=Adam_decoder_RAND_PNP,
                                                       Adam_embedding=Adam_embedding_RAND_PNP,
                                                       flag_clear_target_train=flag_clear_target_train,
                                                       flag_clear_target_test=flag_clear_target_test,
                                                       sigma_out=sigma_out,
                                                       Optim_settings=cfg['Optimizer'], model_name='RAND_PNP'
                                                       )
        # Sync testdata
        Active_RAND.test_input_tensor = test_pool_data_BALD_BALD.clone().detach()
        Active_RAND.test_target_tensor = test_target_data.clone().detach()
        Active_RAND_PNP.test_input_tensor =test_pool_data_BALD_BALD.clone().detach()
        Active_RAND_PNP.test_target_tensor = test_target_data.clone().detach()

        # Train the model[
        W_sample = Active_BALD.train_BNN(observed_train_BALD_rz, eps=args.step_sghmc, max_sample_size=40,
                                         tot_epoch=epoch_orig+500,
                                         thinning=10,
                                         hyper_param_update=25000, sample_int=10,
                                         flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder,
                                         Adam_embedding=Adam_embedding, Drop_p=Drop_orig, list_p_z=list_p_z,
                                         test_input_tensor=test_pool_data_BALD_BALD.clone().detach(),
                                         test_target_tensor=test_target_data.clone().detach(),
                                         conditional_coef=conditional_coef_sghmc,
                                         target_dim=-1, flag_reset_optim=flag_reset_optim,
                                         W_dict_init=W_dict_init, valid_data=valid_data_input_tensor,
                                         valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update)

        # Train the model using RAND
        W_sample_RAND = Active_RAND.train_BNN(observed_train_RAND_rz, eps=args.step_sghmc, max_sample_size=40,
                                              tot_epoch=epoch_orig+500,
                                              thinning=10,
                                              hyper_param_update=25000, sample_int=10,
                                              flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder_RAND,
                                              Adam_embedding=Adam_embedding_RAND, Drop_p=Drop_orig,
                                              list_p_z=list_p_z_RAND,
                                              test_input_tensor=test_pool_data_BALD_BALD.clone().detach(),
                                              test_target_tensor=test_target_data.clone().detach(),
                                              conditional_coef=conditional_coef_sghmc,
                                              target_dim=-1, flag_reset_optim=flag_reset_optim,
                                              W_dict_init=W_dict_init_RAND, valid_data=valid_data_input_tensor,
                                              valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update)
        # Train using PNP RAND
        Active_RAND_PNP.train_BNN(
                                  observed_train=observed_train_RAND_PNP_rz,
                                  KL_coef_W=KL_coef_W, flag_hybrid=flag_hybrid, target_dim=-1,
                                  valid_data=valid_data_input_tensor, valid_data_target=valid_data_target_tensor,conditional_coef=conditional_coef,
                                  **Dict_training_settings_PNP)


        #### Evaluation
        # BALD_BALD
        RMSE_BALD_BALD, MAE_BALD_BALD, NLL_BALD_BALD = test_UCI_AL(model=Active_BALD, max_selection=max_selection_eval,
                                                                   sample_x=None,
                                                                   test_input=test_input_BALD_BALD,
                                                                   test_pool=test_pool_data_BALD_BALD,
                                                                   test_target=test_target_data, sigma_out=sigma_out,
                                                                   search='Target',
                                                                   model_name='PNP_SGHMC', W_sample=W_sample)
        # BALD_RAND
        RMSE_BALD_RAND, MAE_BALD_RAND, NLL_BALD_RAND = test_UCI_AL(model=Active_BALD, max_selection=max_selection_eval,
                                                                   sample_x=None,
                                                                   test_input=test_input_BALD_RAND,
                                                                   test_pool=test_pool_data_BALD_RAND,
                                                                   test_target=test_target_data, sigma_out=sigma_out,
                                                                   search='Random',
                                                                   model_name='PNP_SGHMC', W_sample=W_sample)
        # RAND_BALD
        RMSE_RAND_BALD, MAE_RAND_BALD, NLL_RAND_BALD = test_UCI_AL(model=Active_RAND, max_selection=max_selection_eval,
                                                                   sample_x=None,
                                                                   test_input=test_input_RAND_BALD,
                                                                   test_pool=test_pool_data_RAND_BALD,
                                                                   test_target=test_target_data, sigma_out=sigma_out,
                                                                   search='Target',
                                                                   model_name='PNP_SGHMC',
                                                                   W_sample=W_sample_RAND)

        # RAND_RAND
        RMSE_RAND_RAND, MAE_RAND_RAND, NLL_RAND_RAND = test_UCI_AL(model=Active_RAND, max_selection=max_selection_eval,
                                                                   sample_x=None,
                                                                   test_input=test_input_RAND_RAND,
                                                                   test_pool=test_pool_data_RAND_RAND,
                                                                   test_target=test_target_data, sigma_out=sigma_out,
                                                                   search='Random',
                                                                   model_name='PNP_SGHMC',
                                                                   W_sample=W_sample_RAND)
        # RAND_PNP_BALD
        RMSE_RAND_PNP_BALD, MAE_RAND_PNP_BALD, NLL_RAND_PNP_BALD = test_UCI_AL(model=Active_RAND_PNP,
                                                                               max_selection=max_selection_eval,
                                                                               sample_x=200,
                                                                               test_input=test_input_RAND_PNP_BALD,
                                                                               test_pool=test_pool_data_RAND_PNP_BALD,
                                                                               test_target=test_target_data,
                                                                               sigma_out=sigma_out,
                                                                               search='Target',
                                                                               model_name='PNP_BNN',
                                                                               W_sample=None)
        # PNP_RAND_RAND
        RMSE_RAND_PNP_RAND, MAE_RAND_PNP_RAND, NLL_RAND_PNP_RAND = test_UCI_AL(model=Active_RAND_PNP,
                                                                               max_selection=max_selection_eval,
                                                                               sample_x=200,
                                                                               test_input=test_input_RAND_PNP_RAND,
                                                                               test_pool=test_pool_data_RAND_PNP_RAND,
                                                                               test_target=test_target_data,
                                                                               sigma_out=sigma_out,
                                                                               search='Random',
                                                                               model_name='PNP_BNN',
                                                                               W_sample=None)
        # Store data


        # Compute AUIC
        AUIC_Diff_BALD = Compute_AUIC_1D(const=None, BALD=NLL_BALD_BALD, RAND=NLL_BALD_RAND)
        AUIC_Diff_RAND = Compute_AUIC_1D(const=None, BALD=NLL_RAND_BALD, RAND=NLL_RAND_RAND)
        AUIC_Diff_RAND_PNP = Compute_AUIC_1D(const=None, BALD=NLL_RAND_PNP_BALD, RAND=NLL_RAND_PNP_RAND)
        AUIC_BALD_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_BALD_BALD)
        AUIC_RAND_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_RAND_BALD)
        AUIC_RAND_PNP_BALD = Compute_AUIC_1D(const=-2.0, Results=NLL_RAND_PNP_BALD)
        AUIC_Diff = Compute_AUIC_Differece(AUIC_BALD_BALD, target1=AUIC_RAND_BALD, target2=AUIC_RAND_PNP_BALD)

        Results_list = ['%s' % (counter_loop * step),
                        '%.3f , %.3f , %.3f' % (AUIC_BALD_BALD, AUIC_RAND_BALD, AUIC_RAND_PNP_BALD),
                        '%.3f , %.3f , %.3f' % (AUIC_Diff_BALD, AUIC_Diff_RAND, AUIC_Diff_RAND_PNP)] + AUIC_Diff + \
                       ['%.3f , %.3f' % (BALD_pattern_old, BALD_pattern_new)] + [
                           '%.3f , %.3f' % (RAND_pattern_old, RAND_pattern_new)] + [
                           '%.3f , %.3f' % (RAND_PNP_pattern_old, RAND_PNP_pattern_new)]\
                       + [
                           get_choice(observed_train_BALD - observed_train_BALD_old).tolist()] \
                       + [get_choice(observed_train_BALD)] + [get_choice(test_input_BALD_BALD).tolist()] + [
                           get_choice(test_input_RAND_BALD).tolist()] + [
                           get_choice(test_input_RAND_PNP_BALD).tolist()]+[[BALD_outlier_dim,BALD_outlier_value]]+[[RAND_outlier_dim,RAND_outlier_value]]+[[RAND_PNP_outlier_dim,RAND_PNP_outlier_value]]
        table = Table_format(table, Results_list)
        # Store the table
        # setting_num = 'Debug_4'
        store_Table(filepath_out + '/Setting%s_Seed_%s' % (setting_num, rand_seed) + '.txt',
                    table, title='UCI_%s_Setting_%s_Seed_%s' % (uci_number, setting_num, rand_seed))

        print('Run:%s Data: %s' % (0, counter_loop * step))
        print('NLL_BALD_BALD:%s' % (NLL_BALD_BALD))
        print('NLL_BALD_RAND:%s' % (NLL_BALD_RAND))
        print('NLL_PNP_RAND_BALD:%s' % (NLL_RAND_PNP_BALD))
        print('NLL_PNP_RAND_RAND:%s' % (NLL_RAND_PNP_RAND))
        print('NLL_RAND_BALD:%s' % (NLL_RAND_BALD))
        print('NLL_RAND_RAND:%s' % (NLL_RAND_RAND))
        print(
            'AUIC_Diff_BALD: %.3f RAND: %.3f RAND_PNP: %.3f' % (AUIC_Diff_BALD, AUIC_Diff_RAND, AUIC_Diff_RAND_PNP))
        print('AUIC BALD: %.3f RAND: %.3f RAND_PNP: %.3f' % (AUIC_BALD_BALD, AUIC_RAND_BALD, AUIC_RAND_PNP_BALD))
        # Store the results
        RMSE_BALD_mat_BALD[runs, counter_loop, :], MAE_BALD_mat_BALD[runs, counter_loop, :], NLL_BALD_mat_BALD[runs,
                                                                                             counter_loop,
                                                                                             :] = RMSE_BALD_BALD, MAE_BALD_BALD, NLL_BALD_BALD
        RMSE_BALD_mat_RAND[runs, counter_loop, :], MAE_BALD_mat_RAND[runs, counter_loop, :], NLL_BALD_mat_RAND[runs,
                                                                                             counter_loop,
                                                                                             :] = RMSE_BALD_RAND, MAE_BALD_RAND, NLL_BALD_RAND
        RMSE_RAND_mat_BALD[runs, counter_loop, :], MAE_RAND_mat_BALD[runs, counter_loop, :], NLL_RAND_mat_BALD[runs,
                                                                                             counter_loop,
                                                                                             :] = RMSE_RAND_BALD, MAE_RAND_BALD, NLL_RAND_BALD
        RMSE_RAND_mat_RAND[runs, counter_loop, :], MAE_RAND_mat_RAND[runs, counter_loop, :], NLL_RAND_mat_RAND[runs,
                                                                                             counter_loop,
                                                                                             :] = RMSE_RAND_RAND, MAE_RAND_RAND, NLL_RAND_RAND
        RMSE_RAND_PNP_mat_BALD[runs, counter_loop, :], MAE_RAND_PNP_mat_BALD[runs, counter_loop,
                                                       :], NLL_RAND_PNP_mat_BALD[runs, counter_loop,
                                                           :] = RMSE_RAND_PNP_BALD, MAE_RAND_PNP_BALD, NLL_RAND_PNP_BALD
        RMSE_RAND_PNP_mat_RAND[runs, counter_loop, :], MAE_RAND_PNP_mat_RAND[runs, counter_loop,
                                                       :], NLL_RAND_PNP_mat_RAND[runs, counter_loop,
                                                           :] = RMSE_RAND_PNP_RAND, MAE_RAND_PNP_RAND, NLL_RAND_PNP_RAND

        # Store Data
        # if counter_selection % 200==0:
        #     if Select_Split == True:
        #         filename_2 = 'Debug_UCI%s_Step%s_total%s_%s_CTTest_Conditional%s_%sBALD_%sBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_WeightTar1_eps%s_Adam%s_ProbSelect_ALTest_Opposite_StoredData_'\
        #                      % (uci_number,step,max_selection,args.uci_normal,conditional_coef,BALD_coef,balance_coef,sigma_out,Drop_orig,args.step_sghmc,args.lr_sghmc)
        #     else:
        #         filename_2 = 'Debug_UCI%s_Step%s_total%s_%s_CTTest_Conditional%s_%sBALD_NoBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_WeightTar1_eps%s_Adam%s_ProbSelect_ALTest_Opposite_StoredData_' \
        #                      % (uci_number, step, max_selection, args.uci_normal, conditional_coef, BALD_coef,sigma_out, Drop_orig, args.step_sghmc, args.lr_sghmc)
        #     observed_train_BALD_rz_np = observed_train_BALD_rz.cpu().data.numpy()
        #     observed_train_RAND_rz_np = observed_train_RAND_rz.cpu().data.numpy()
        #     np.save(filepath_out + '/%sBALD_datanum_%s_Seed_%s.npy' % (filename_2,counter_loop * step,rand_seed), observed_train_BALD_rz_np)
        #     np.save(filepath_out + '/%sRAND_datanum_%s_Seed_%s.npy' % (filename_2,counter_loop * step,rand_seed), observed_train_RAND_rz_np)

        counter_loop+=1
        if idx_end>observed_train_BALD.shape[0]:
            idx_start=0
            idx_end=idx_start+batch_select
        else:
            idx_start=idx_end
            idx_end=idx_start+batch_select
        print('Shape:%s'%(RMSE_BALD_BALD.shape))
        if Select_Split==True:
            filename_2 = 'Debug_UCI%s_Step%s_total%s_%s_CTTest_Conditional%s_%sBALD_%sBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_WeightTar1_eps%s_Adam%s_ProbSelect_ALTest_%sResults_' \
                     % (uci_number, step, max_selection, args.uci_normal, conditional_coef, BALD_coef, balance_coef,
                        sigma_out, Drop_orig, args.step_sghmc, args.lr_sghmc,suffix)
        else:
            filename_2 = 'Debug_UCI%s_Step%s_total%s_%s_CTTest_Conditional%s_%sBALD_NoBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_WeightTar1_eps%s_Adam%s_ProbSelect_ALTest_%sResults_' \
                         % (uci_number, step, max_selection, args.uci_normal, conditional_coef, BALD_coef,
                            sigma_out, Drop_orig, args.step_sghmc, args.lr_sghmc,suffix)
        #Store the results
        BALD_BALD_Result=Results_Class(RMSE_test=RMSE_BALD_mat_BALD,MAE_test=MAE_BALD_mat_BALD,NLL_test=NLL_BALD_mat_BALD)
        BALD_RAND_Result=Results_Class(RMSE_test=RMSE_BALD_mat_RAND,MAE_test=MAE_BALD_mat_RAND,NLL_test=NLL_BALD_mat_RAND)

        RAND_BALD_Result=Results_Class(RMSE_test=RMSE_RAND_mat_BALD,MAE_test=MAE_RAND_mat_BALD,NLL_test=NLL_RAND_mat_BALD)
        RAND_RAND_Result=Results_Class(RMSE_test=RMSE_RAND_mat_RAND,MAE_test=MAE_RAND_mat_RAND,NLL_test=NLL_RAND_mat_RAND)

        PNP_RAND_BALD_Result=Results_Class(RMSE_test=RMSE_RAND_PNP_mat_BALD,MAE_test=MAE_RAND_PNP_mat_BALD,NLL_test=NLL_RAND_PNP_mat_BALD)
        PNP_RAND_RAND_Result=Results_Class(RMSE_test=RMSE_RAND_PNP_mat_RAND,MAE_test=MAE_RAND_PNP_mat_RAND,NLL_test=NLL_RAND_PNP_mat_RAND)


        filename = filepath_out
        save_data(filepath_out + '/Stored_Results/%s'%(filename_2)+'BALD_BALD.pkl', BALD_BALD_Result)
        save_data(filepath_out + '/Stored_Results/%s'%(filename_2)+'BALD_RAND.pkl', BALD_RAND_Result)

        save_data(filepath_out + '/Stored_Results/%s'%(filename_2)+'RAND_BALD.pkl', RAND_BALD_Result)
        save_data(filepath_out + '/Stored_Results/%s'%(filename_2)+'RAND_RAND.pkl', RAND_RAND_Result)

        save_data(filepath_out + '/Stored_Results/%s'%(filename_2)+'PNP_RAND_BALD.pkl', PNP_RAND_BALD_Result)
        save_data(filepath_out + '/Stored_Results/%s'%(filename_2)+'PNP_RAND_RAND.pkl', PNP_RAND_RAND_Result)


