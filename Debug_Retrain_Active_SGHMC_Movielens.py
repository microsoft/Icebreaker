'''
This file implements the Active training by retraining the PNP model from scratch using SGHMC
'''

import sys
sys.path.append('/mnt/_default/SEDDI_Code/')
sys.path.append('/mnt/_default/SEDDI_Code/base_Model')
sys.path.append('/mnt/_default/SEDDI_Code/Dataloader')
sys.path.append('/mnt/_default/SEDDI_Code/Util')

from base_Model.base_Network import *
import os

import argparse

from Dataloader.base_Dataloader import *
from Dataloader.base_mask import *
from torch.utils.data import DataLoader
from base_Model.base_BNN import *
from base_Model.BNN_Network_zoo import *
import copy
from torch.autograd import grad
from scipy.stats import bernoulli
from base_Model.base_Active_Learning import *
from base_Model.base_Active_Learning_SGHMC import *
import random



torch.set_default_tensor_type('torch.cuda.FloatTensor')


parser = argparse.ArgumentParser(description='SEDDI Active Learning Imputation')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr_Adam', type=float, default=0.003, metavar='LR_ADAM',
                    help='learning rate for Adam (default: 0.003)')
parser.add_argument('--lr_sghmc', type=float, default=0.003, metavar='LR_SGHMC_ADAM',
                    help='learning rate for sghmc Adam (default: 0.003)')
parser.add_argument('--step_sghmc', type=float, default=3e-4, metavar='LR_SGHMC',
                    help='step size for sghmc (default: 3e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed list (default: 1)')

parser.add_argument('--sigma', type=float, default=0.02, metavar='SIGMA',
                    help='What sigma output should be used default:0.02')

parser.add_argument('--BALD_Coef', type=float, default=1, metavar='BALD',
                    help='What is the BALD Coef used Default: 1 For MC, do not change this!')

parser.add_argument('--Conditional_coef', type=float, default=1, metavar='COND',
                    help='The Conditional coef for sghmc and pnp. For MC do not change this')

parser.add_argument('--scale_data', type=float, default=1, metavar='SCALE',
                    help='Scale data with default 1')


parser.add_argument('--noisy_update', type=float, default=1., metavar='NOISY',
                    help='whether perform noisy optimization default: Noisy update')

# output dir
parser.add_argument('--output_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', '/tmp'))

args = parser.parse_args()


# Meta hyperparameter




# Overwrite by parser

rand_seed_list=[args.seed]
num_runs=1



filepath_out = args.output_dir




# Set random seed
rand_seed = rand_seed_list[0]
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)




# Define the base dataset, this will be overwritten if use stored dataset
filename='/mnt/_default/dataset/movielens/Fixed_overall_mat.pkl'
movielens1m_mat=load_data(filename)
movielens1m_mat=torch.from_numpy(movielens1m_mat).float()
overall_data=Select_Movielens(movielens1m_mat,size1=1000,size2=2000).cpu().data.numpy()

torch.backends.cudnn.deterministic = True




################################ Meta settings

Select_Split=True
Strategy='MC'

selection_scheme='overall'
##### Always False #####
flag_LV=False
########################

batch_select=200

scale_data=1.
noisy_update=1.

flag_minibatch=False ########## DELETE THIS

flag_stored_dataset=True

flag_prob_select=False ######## Delete this

if selection_scheme=='per_user' or selection_scheme=='batch':
    Select_Split=False

if flag_LV:
    suffix='%s_%s_%s_'%(selection_scheme,Strategy,'LV')
else:
    suffix='%s_%s_'%(selection_scheme,Strategy)
###################################

# Read cfg file
code_path=os.getenv('PT_CODE_DIR')
print('code_path:%s'%(code_path))
cfg=ReadYAML('/mnt/_default/SEDDI_Code/Config/Config_SGHMC_PNP_movielens1m.yaml')


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
obs_dim=overall_data.shape[1]
latent_dim=cfg['BNN_Settings']['latent_dim']
dim_before_agg=cfg['BNN_Settings']['dim_before_agg']
embedding_dim=cfg['BNN_Settings']['embedding_dim']
add_const=cfg['BNN_Settings']['decoder_settings']['add_const']
flag_log_q=cfg['BNN_Settings']['flag_log_q']
flag_clear_target_train=cfg['Active_Learning_Settings']['flag_clear_target_train']
flag_clear_target_test=cfg['Active_Learning_Settings']['flag_clear_target_test']
flag_stream=cfg['Training_Settings']['flag_stream']
flag_reset_variance=cfg['Active_Learning_Settings']['flag_reset_variance']
flag_hybrid=cfg['Active_Learning_Settings']['flag_hybrid']
conditional_coef=1.
balance_coef=cfg['Active_Learning_Settings']['balance_coef']
explore_coef=cfg['Active_Learning_Settings']['explore_coef']
BALD_coef=1.
# KL Schedule
KL_Schedule_Settings=cfg['KL_Schedule_Settings']
KL_coef_W=KL_Schedule_helper(**KL_Schedule_Settings)
# KL Pretrain_Schedule
KL_Schedule_Pretrain_Settings=cfg['KL_Schedule_Pretrain_Settings']
KL_coef_W_pretrain=KL_Schedule_helper(**KL_Schedule_Pretrain_Settings)

max_selection=cfg['Active_Learning_Settings']['max_selection'] # This for active learning of W
flag_pretrain=cfg['Pretrain_Settings']['flag_pretrain']
step = cfg['Active_Learning_Settings']['step']

# Store the Results
# TODO: Store the results
RMSE_BALD_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))
MAE_BALD_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))
NLL_BALD_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))

RMSE_RAND_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))
MAE_RAND_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))
NLL_RAND_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))

RMSE_RAND_PNP_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))
MAE_RAND_PNP_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))
NLL_RAND_PNP_mat=np.zeros((num_runs,math.ceil(max_selection/step)+1))


cfg['Optimizer']['lr_sghmc']=args.lr_sghmc
cfg['Optimizer']['lr']=args.lr_Adam

for runs in range(num_runs):
    counter_selection = 0

    ########### ########## ########## ########## ########## ########## ########## Pretrain the model ########## ########## ########## ########## ########## ########## ##########
    # Setup table
    table = PrettyTable()
    table.field_names = ['Data', 'NLL', 'NLL_Diff_R', 'NLL_Diff_P', 'SP_BALD', 'SP_RAND', 'SP_RANDP', 'Max selection',
                         'SP Num','RAND SP Num']

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
                                         pooling_act=pooling_act, flag_log_q=flag_log_q,flag_LV=flag_LV)
    PNP_SGHMC_RAND = Point_Net_Plus_BNN_SGHMC(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                              encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                              encoder_hidden_before_agg=encoder_hidden_before_agg, \
                                              encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                              encoder_hidden_after_agg=encoder_hidden_after_agg,
                                              embedding_dim=embedding_dim,
                                              decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden, \
                                              pooling=pooling, output_const=output_const, add_const=add_const,
                                              sample_z=sample_z, sample_W=sample_W, W_sigma_prior=W_sigma_prior,
                                              pooling_act=pooling_act, flag_log_q=flag_log_q,flag_LV=flag_LV)

    PNP_Det = Point_Net_Plus_BNN(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                 encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                 encoder_hidden_before_agg=encoder_hidden_before_agg,
                                 encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                 encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                 decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden,
                                 pooling=pooling, output_const=output_const, add_const=add_const, sample_z=sample_z,
                                 sample_W=sample_W_PNP, W_sigma_prior=W_sigma_prior, pooling_act=pooling_act,
                                 BNN_init_range=BNN_init_range, BNN_coef_sample=0.,
                                 KL_coef=KL_coef, flag_local=True, couple_decoder_encoder=False, flag_log_q=flag_log_q,flag_LV=flag_LV
                                 )

    ##################### Get state dict ##################### ##################### ##################### #####################
    decoder_state_SGHMC, encoder_before_state_SGHMC, encoder_after_state_SGHMC, \
    embedding_state_SGHMC, embedding_bias_SGHMC = PNP_SGHMC._extract_state_dict()
    decoder_state_Det, encoder_before_state_Det, encoder_after_state_Det, \
    embedding_state_Det, embedding_bias_Det = PNP_Det._extract_state_dict()

    #####################################################################################################################################
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
    # Load AL eval Settings
    sample_x_eval = cfg['AL_Eval_Settings']['sample_x']

    # Load other settings
    Dict_training_settings = cfg['Training_Settings']
    Dict_training_settings_RAND = copy.deepcopy(Dict_training_settings)
    Dict_training_settings_PNP = copy.deepcopy(Dict_training_settings)
    Dict_training_settings_PNP['flag_BNN'] = False  # This is to turn off the BNN training for PNP
    Dict_training_settings_PNP['flag_stream'] = False

    Dict_dataset_settings = cfg['Dataset_Settings']

    # DEFINE Active Learning obj
    Active_BALD = base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC,Infer_model=Infer_SGHMC,overall_data=overall_data,rs=rand_seed,sigma_out=sigma_out,Optim_settings=cfg['Optimizer'],Adam_encoder=Adam_encoder,
                                                 Adam_embedding=Adam_embedding,flag_clear_target_train=flag_clear_target_train,flag_clear_target_test=flag_clear_target_test,model_name='SGHMC Active')
    Active_RAND=base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC_RAND,Infer_model=Infer_SGHMC_RAND,overall_data=overall_data,rs=rand_seed,sigma_out=sigma_out,Optim_settings=cfg['Optimizer'],Adam_encoder=Adam_encoder_RAND,
                                                 Adam_embedding=Adam_embedding_RAND,flag_clear_target_train=flag_clear_target_train,flag_clear_target_test=flag_clear_target_test,model_name='SGHMC Active RAND')

    Active_RAND_PNP = base_Active_Learning_Decoder(model=PNP_Det, overall_data=overall_data, active_scheme='1',
                                                   rs=rand_seed, Adam_encoder=Adam_encoder_RAND_PNP,
                                                   Adam_decoder=Adam_decoder_RAND_PNP,
                                                   Adam_embedding=Adam_embedding_RAND_PNP,
                                                   flag_clear_target_train=flag_clear_target_train,
                                                   flag_clear_target_test=flag_clear_target_test,
                                                   sigma_out=sigma_out,
                                                   Optim_settings=cfg['Optimizer'], model_name='RAND_PNP'
                                                   )

    # Pre-Process the data (Only Define Once)
    if flag_stored_dataset==False:
        Active_BALD._data_preprocess(test_missing=0.2, **Dict_dataset_settings)
    else:
        filenamepath = '/mnt/_default/dataset/movielens/Seed_5_'
        Active_BALD._load_data(filepathname=filenamepath)



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
    Drop_p=0.2
    Dict_training_settings_PNP['Drop_p'] = Drop_p

    epoch_orig = args.epochs
    tot_epoch=args.epochs
    Dict_training_settings_PNP['epoch'] = tot_epoch
    Dict_training_settings_PNP['sigma_out']=sigma_out
    conditional_coef=1.
    conditional_coef_sghmc=1.

    # Load the stored data for debug purpose
    # filename_data = cfg['File_Settings']['Result_path']
    # #pretrain_data=np.load(filename_data+'Stored_Data/Active_Learning_W_UCI1_SGHMC_ep2000_step50_total1500_CTTest_Conditional0.8_BALD0.92_Balance0.5_Temp0.25p0_Drop0.2_Sigma0.3_UpdateEncoder_WeightTar1_eps3e-5_Adam0.003_ProbSelect_ALTest_BALD_datanum_1000_Seed_730.npy')
    # pretrain_data=np.load(filename_data+'Stored_Data/Active_Learning_W_UCI1_SGHMC_ep2000_step25_total1000_CTTest_Conditional0.8_BALD0.92_Balance0.5_Temp0.25p0_Drop0.2_Sigma0.4_UpdateEncoder_WeightTar1_eps5e-5_Adam0.003_ProbSelect_Alpha_ALTest_BALD_datanum_1000_Seed_730.npy')
    # pretrain_data=torch.from_numpy(pretrain_data).float().cuda()
    #
    # # Pretrain
    Active_RAND_PNP.train_BNN(counter_loop=0, max_selection=max_selection, flag_pretrain=True,
                              observed_train=pretrain_data,
                              random_seed=rand_seed, KL_coef_W=KL_coef_W_pretrain, flag_hybrid=flag_hybrid,
                              target_dim=-1,valid_data=valid_data_input_tensor,valid_data_target=valid_data_target_tensor,
                              **Dict_training_settings_PNP)





    W_sample_RAND = Active_RAND.train_BNN(pretrain_data, eps=args.step_sghmc, max_sample_size=40, tot_epoch=tot_epoch+0,
                                          thinning=2,
                                          hyper_param_update=25000, flag_reset_r=False, sample_int=10,
                                          flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder_RAND,
                                          Adam_embedding=Adam_embedding_RAND, Drop_p=Drop_p,
                                          list_p_z=list_p_z_RAND,
                                          test_input_tensor=Active_RAND.test_input_tensor,
                                          test_target_tensor=Active_RAND.test_target_tensor, conditional_coef=conditional_coef_sghmc,
                                          target_dim=-1,flag_reset_optim=flag_reset_optim,valid_data=valid_data_input_tensor,valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update)

    W_sample = Active_BALD.train_BNN(pretrain_data, eps=args.step_sghmc, max_sample_size=40, tot_epoch=tot_epoch+0,
                                     thinning=2,
                                     hyper_param_update=25000, flag_reset_r=False, sample_int=10,
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
    Dict_training_settings_PNP['epoch'] = epoch_orig
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

    train_data_perm = Active_BALD.train_data_tensor.clone().detach()

    test_input_perm = Active_BALD.test_input_tensor.clone().detach()

    test_target_data = Active_BALD.test_target_tensor.clone().detach()

    # Initialize test input
    test_input_BALD = test_input_perm.clone().detach()  # N_test x obs_dim
    test_input_RAND = test_input_perm.clone().detach()
    test_input_RAND_PNP = test_input_perm.clone().detach()




    ############################################################################################################################################################


    ################################################### Evaluation of the pretrain model ########################################################################
    RMSE_RAND_PNP, MAE_RAND_PNP, NLL_RAND_PNP = Test_UCI_batch(PNP_Det, test_input_RAND_PNP, test_target_data,
                                                               sigma_out_scale=sigma_out, flag_fixed=True, split=10,
                                                               flag_model='PNP_BNN', size=10, Infer_model=None,
                                                               W_sample=None)

    RMSE_RAND, MAE_RAND, NLL_RAND = Test_UCI_batch(PNP_SGHMC_RAND, test_input_RAND, test_target_data,
                                                   sigma_out_scale=sigma_out, flag_fixed=True, split=10,
                                                   flag_model='PNP_SGHMC', size=10, Infer_model=Infer_SGHMC_RAND,
                                                   W_sample=W_sample_RAND)

    RMSE_BALD, MAE_BALD, NLL_BALD = Test_UCI_batch(PNP_SGHMC, test_input_BALD, test_target_data,
                                                   sigma_out_scale=sigma_out, flag_fixed=True, split=10,
                                                   flag_model='PNP_SGHMC', size=10, Infer_model=Infer_SGHMC,
                                                   W_sample=W_sample)


    # Compute Diff
    NLL_diff_R = NLL_RAND.cpu().data.numpy() - NLL_BALD.cpu().data.numpy()
    NLL_diff_P = NLL_RAND_PNP.cpu().data.numpy() - NLL_BALD.cpu().data.numpy()

    Results_list = ['%s Init' % (0 * step),
                    '%.3f , %.3f , %.3f' % (
                        NLL_BALD.cpu().data.numpy(), NLL_RAND.cpu().data.numpy(), NLL_RAND_PNP.data.cpu().numpy())] + [
                       NLL_diff_R, NLL_diff_P] + \
                   ['Full', 'Full', 'Full', 'None', 'None','None']  # + \
    # [get_choice(observed_train_BALD).tolist()] \
    # + [get_choice(observed_train_BALD).tolist()] + [get_choice(observed_train_RAND)] + [
    #     get_choice(observed_train_RAND_PNP).tolist()]
    table = Table_format(table, Results_list)
    # Store the table
    if Select_Split==True:
        setting_num = 'Debug_Step%s_total%s_NCTTest_Conditional%s_%sBALD_%sBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_eps%s_Adam%s_MC_%sPTable_'\
    %(step,max_selection,conditional_coef,BALD_coef,balance_coef,sigma_out,Drop_orig,args.step_sghmc,args.lr_sghmc,suffix)
    else:
        print('Indeed No Balance')
        setting_num = 'Debug_Step%s_total%s_NCTTest_Conditional%s_%sBALD_NoBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_eps%s_Adam%s_MC_%sPTable_' \
                      % (step, max_selection, conditional_coef, BALD_coef, sigma_out,
                         Drop_orig, args.step_sghmc, args.lr_sghmc,suffix)

    print('Run:pretrain Data: %s' % (counter_loop * step))
    print('NLL_BALD:%s' % (NLL_BALD.cpu().data.numpy()))
    print('NLL_RAND:%s' % (NLL_RAND.cpu().data.numpy()))
    print('NLL_RAND_PNP:%s' % (NLL_RAND_PNP.cpu().data.numpy()))
    # Store Results
    store_Table(filepath_out + '/Setting%s_Seed_%s' % (setting_num, rand_seed) + '.txt',
                table, title='Movielens_Setting_%s_Seed_%s' % (setting_num, rand_seed))

    RMSE_BALD_mat[runs, 0], MAE_BALD_mat[runs, 0], NLL_BALD_mat[runs, 0], RMSE_RAND_mat[runs, 0], MAE_RAND_mat[runs, 0], \
    NLL_RAND_mat[runs, 0], \
    RMSE_RAND_PNP_mat[runs, 0], MAE_RAND_PNP_mat[runs, 0], NLL_RAND_PNP_mat[
        runs, 0] = RMSE_BALD.cpu().data.numpy(), MAE_BALD.cpu().data.numpy(), NLL_BALD.cpu().data.numpy(), \
                   RMSE_RAND.cpu().data.numpy(), MAE_RAND.cpu().data.numpy(), NLL_RAND.cpu().data.numpy(), \
                   RMSE_RAND_PNP.cpu().data.numpy(), MAE_RAND_PNP.cpu().data.numpy(), NLL_RAND_PNP.cpu().data.numpy()

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

        # Record old train data
        observed_train_BALD_old = observed_train_BALD.clone().detach()
        observed_train_RAND_old = observed_train_RAND.clone().detach()
        observed_train_RAND_PNP_old = observed_train_RAND_PNP.clone().detach()
        # Initialization for test data

        test_input_BALD = test_input_perm.clone().detach()  # N_test x obs_dim
        test_input_RAND = test_input_perm.clone().detach()
        test_input_RAND_PNP = test_input_perm.clone().detach()


        # Active selection BALD
        if counter_loop <= 0:
            observed_train_BALD, train_pool_data_BALD, flag_full, num_selected = Active_BALD.base_random_select_training(
                observed_train=observed_train_BALD, pool_data=train_pool_data_BALD, step=num_selected_variable,
                sigma_out=sigma_out, eps=explore_coef, flag_initial=flag_init_train, balance_prop=balance_coef,Select_Split=Select_Split,selection_scheme=selection_scheme,\
            idx_start=idx_start,idx_end=idx_end,split=200,flag_minibatch=flag_minibatch)

        else:
            # coef_explor=1./(1+math.exp(0.1*(counter_loop-10)))
            # coef_explor=np.minimum(0.7,np.maximum(coef_explor,0.4))
            # coef_explor=0.3
            # print('Coef:%s'%(coef_explor))
            flag_weight=int((counter_loop+1)%2)
            observed_train_BALD, train_pool_data_BALD, flag_full, num_selected = Active_BALD.base_active_learning_decoder(
                eps=explore_coef, balance_prop=balance_coef, coef=BALD_coef,
                observed_train=observed_train_BALD, pool_data=train_pool_data_BALD, step=num_selected_variable,
                active_sample_number=sample_x, flag_initial=flag_init_train, sigma_out=sigma_out, W_sample=W_sample,strategy=Strategy,strategy_alternating=flag_weight,Select_Split=Select_Split,selection_scheme=selection_scheme,\
            idx_start=idx_start,idx_end=idx_end,flag_prob_select=flag_prob_select,flag_minibatch=flag_minibatch)


            # observed_train_BALD,train_pool_data_BALD=Active_BALD.base_random_select_row(observed_train=observed_train_BALD,pool_data=train_pool_data_BALD,step=num_selected_variable)

        observed_train_BALD = observed_train_BALD.clone().detach()
        train_pool_data_BALD = train_pool_data_BALD.clone().detach()
        # Rand Selection
        observed_train_RAND, train_pool_data_RAND, flag_full, num_selected = Active_RAND.base_random_select_training(
            observed_train=observed_train_RAND, pool_data=train_pool_data_RAND, step=num_selected_variable,
            sigma_out=sigma_out, eps=explore_coef, balance_prop=balance_coef, flag_initial=flag_init_train,Select_Split=Select_Split,selection_scheme=selection_scheme,\
            idx_start=idx_start,idx_end=idx_end,flag_minibatch=flag_minibatch)

        observed_train_RAND = observed_train_RAND.clone().detach()
        train_pool_data_RAND = train_pool_data_RAND.clone().detach()
        # Rand PNP Selection
        observed_train_RAND_PNP, train_pool_data_RAND_PNP, flag_full, num_selected = Active_RAND_PNP.base_random_select_training(
            observed_train=observed_train_RAND_PNP, pool_data=train_pool_data_RAND_PNP, step=num_selected_variable,
            sigma_out=sigma_out, eps=explore_coef,
            balance_prop=balance_coef, flag_initial=flag_init_train,Select_Split=Select_Split,selection_scheme=selection_scheme,\
            idx_start=idx_start,idx_end=idx_end
        )

        observed_train_RAND_PNP = observed_train_RAND_PNP.clone().detach()
        train_pool_data_RAND_PNP = train_pool_data_RAND_PNP.clone().detach()

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

        if flag_init_train:
            String_BALD = new_selection_num(observed_train_BALD, observed_train_BALD_old)
            String_RAND=new_selection_num(observed_train_RAND, observed_train_RAND_old)
        else:
            String_BALD = old_selection_num(observed_train_BALD, observed_train_BALD_old)
            String_RAND=old_selection_num(observed_train_RAND, observed_train_RAND_old)

        # # Now detect the outlier
        # BALD_outlier_dim, BALD_outlier_value = outlier_detection(observed_train_BALD, observed_train_BALD_old,
        #                                                          diff_scale=1.5)
        # RAND_outlier_dim, RAND_outlier_value = outlier_detection(observed_train_RAND, observed_train_RAND_old,
        #                                                          diff_scale=1.5)
        # RAND_PNP_outlier_dim, RAND_PNP_outlier_value = outlier_detection(observed_train_RAND_PNP,
        #                                                                  observed_train_RAND_PNP_old, diff_scale=1.5)

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
                                             pooling_act=pooling_act, flag_log_q=flag_log_q,flag_LV=flag_LV)
        PNP_SGHMC_RAND = Point_Net_Plus_BNN_SGHMC(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                                  encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                                  encoder_hidden_before_agg=encoder_hidden_before_agg, \
                                                  encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                                  encoder_hidden_after_agg=encoder_hidden_after_agg,
                                                  embedding_dim=embedding_dim,
                                                  decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden, \
                                                  pooling=pooling, output_const=output_const, add_const=add_const,
                                                  sample_z=sample_z, sample_W=sample_W, W_sigma_prior=W_sigma_prior,
                                                  pooling_act=pooling_act, flag_log_q=flag_log_q,flag_LV=flag_LV)

        PNP_Det = Point_Net_Plus_BNN(latent_dim=latent_dim, obs_dim=obs_dim, dim_before_agg=dim_before_agg,
                                     encoder_layer_num_before_agg=encoder_layer_num_before_agg,
                                     encoder_hidden_before_agg=encoder_hidden_before_agg,
                                     encoder_layer_num_after_agg=encoder_layer_num_after_agg,
                                     encoder_hidden_after_agg=encoder_hidden_after_agg, embedding_dim=embedding_dim,
                                     decoder_layer_num=decoder_layer_num, decoder_hidden=decoder_hidden,
                                     pooling=pooling, output_const=output_const, add_const=add_const, sample_z=sample_z,
                                     sample_W=sample_W_PNP, W_sigma_prior=W_sigma_prior, pooling_act=pooling_act,
                                     BNN_init_range=BNN_init_range, BNN_coef_sample=0.,
                                     KL_coef=KL_coef, flag_local=True, couple_decoder_encoder=False, flag_log_q=flag_log_q,flag_LV=flag_LV
                                     )

        ################### Load State Dict
        PNP_SGHMC._load_state_dict(decoder_state_SGHMC, encoder_before_state_SGHMC, encoder_after_state_SGHMC
                                   , embedding_state_SGHMC, embedding_bias_SGHMC)
        PNP_SGHMC_RAND._load_state_dict(decoder_state_SGHMC, encoder_before_state_SGHMC, encoder_after_state_SGHMC
                                        , embedding_state_SGHMC, embedding_bias_SGHMC)
        PNP_Det._load_state_dict(decoder_state_Det, encoder_before_state_Det, encoder_after_state_Det
                                 , embedding_state_Det, embedding_bias_Det)


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
                                                         overall_data=overall_data, rs=rand_seed, sigma_out=sigma_out,
                                                         Optim_settings=cfg['Optimizer'], Adam_encoder=Adam_encoder,
                                                         Adam_embedding=Adam_embedding, flag_clear_target_train=flag_clear_target_train,flag_clear_target_test=flag_clear_target_test,
                                                         model_name='SGHMC Active')
        Active_RAND = base_Active_Learning_SGHMC_Decoder(model=PNP_SGHMC_RAND, Infer_model=Infer_SGHMC_RAND,
                                                         overall_data=overall_data, rs=rand_seed, sigma_out=sigma_out,
                                                         Optim_settings=cfg['Optimizer'], Adam_encoder=Adam_encoder_RAND,
                                                         Adam_embedding=Adam_embedding_RAND,
                                                         flag_clear_target_train=flag_clear_target_train,
                                                         flag_clear_target_test=flag_clear_target_test,
                                                         model_name='SGHMC Active RAND')

        Active_RAND_PNP = base_Active_Learning_Decoder(model=PNP_Det, overall_data=overall_data, active_scheme='1',
                                                       rs=rand_seed, Adam_encoder=Adam_encoder_RAND_PNP,
                                                       Adam_decoder=Adam_decoder_RAND_PNP,
                                                       Adam_embedding=Adam_embedding_RAND_PNP,
                                                       flag_clear_target_train=flag_clear_target_train,
                                                       flag_clear_target_test=flag_clear_target_test,
                                                       sigma_out=sigma_out,
                                                       Optim_settings=cfg['Optimizer'], model_name='RAND_PNP'
                                                       )
        # Sync testdata
        Active_RAND.test_input_tensor = test_input_BALD.clone().detach()
        Active_RAND.test_target_tensor = test_target_data.clone().detach()
        Active_RAND_PNP.test_input_tensor =test_input_BALD.clone().detach()
        Active_RAND_PNP.test_target_tensor = test_target_data.clone().detach()

        # Train the model[
        W_sample = Active_BALD.train_BNN(observed_train_BALD_rz, eps=args.step_sghmc, max_sample_size=40,
                                         tot_epoch=epoch_orig,
                                         thinning=2,
                                         hyper_param_update=25000, flag_reset_r=False, sample_int=10,
                                         flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder,
                                         Adam_embedding=Adam_embedding, Drop_p=Drop_orig, list_p_z=list_p_z,
                                         test_input_tensor=test_input_BALD.clone().detach(),
                                         test_target_tensor=test_target_data.clone().detach(),
                                         conditional_coef=conditional_coef_sghmc,
                                         target_dim=-1, flag_reset_optim=flag_reset_optim,
                                         W_dict_init=W_dict_init, valid_data=valid_data_input_tensor,
                                         valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update,flag_record_best=False)

        # Train the model using RAND
        W_sample_RAND = Active_RAND.train_BNN(observed_train_RAND_rz, eps=args.step_sghmc, max_sample_size=40,
                                              tot_epoch=epoch_orig,
                                              thinning=2,
                                              hyper_param_update=25000, flag_reset_r=False, sample_int=10,
                                              flag_hybrid=flag_hybrid, Adam_encoder=Adam_encoder_RAND,
                                              Adam_embedding=Adam_embedding_RAND, Drop_p=Drop_orig,
                                              list_p_z=list_p_z_RAND,
                                              test_input_tensor=test_input_BALD.clone().detach(),
                                              test_target_tensor=test_target_data.clone().detach(),
                                              conditional_coef=conditional_coef_sghmc,
                                              target_dim=-1, flag_reset_optim=flag_reset_optim,
                                              W_dict_init=W_dict_init_RAND, valid_data=valid_data_input_tensor,
                                              valid_data_target=valid_data_target_tensor,sigma_out=sigma_out,scale_data=args.scale_data,noisy_update=args.noisy_update,
                                              flag_record_best=False)
        # Train using PNP RAND
        Active_RAND_PNP.train_BNN(counter_loop=counter_loop, max_selection=max_selection,
                                  observed_train=observed_train_RAND_PNP_rz,
                                  random_seed=rand_seed,
                                  KL_coef_W=KL_coef_W, flag_hybrid=flag_hybrid, target_dim=-1, flag_init_train=flag_init_train,
                                  valid_data=valid_data_input_tensor, valid_data_target=valid_data_target_tensor,
                                  **Dict_training_settings_PNP)


        #### Evaluation
        RMSE_RAND_PNP, MAE_RAND_PNP, NLL_RAND_PNP = Test_UCI_batch(PNP_Det, test_input_RAND_PNP, test_target_data,
                                                                   sigma_out_scale=sigma_out, flag_fixed=True, split=10,
                                                                   flag_model='PNP_BNN', size=10, Infer_model=None,
                                                                   W_sample=None)

        RMSE_RAND, MAE_RAND, NLL_RAND = Test_UCI_batch(PNP_SGHMC_RAND, test_input_RAND, test_target_data,
                                                       sigma_out_scale=sigma_out, flag_fixed=True, split=10,
                                                       flag_model='PNP_SGHMC', size=10, Infer_model=Infer_SGHMC_RAND,
                                                       W_sample=W_sample_RAND)

        RMSE_BALD, MAE_BALD, NLL_BALD = Test_UCI_batch(PNP_SGHMC, test_input_BALD, test_target_data,
                                                       sigma_out_scale=sigma_out, flag_fixed=True, split=10,
                                                       flag_model='PNP_SGHMC', size=10, Infer_model=Infer_SGHMC,
                                                       W_sample=W_sample)
        # Store data

        NLL_diff_R = NLL_RAND.cpu().data.numpy() - NLL_BALD.cpu().data.numpy()
        NLL_diff_P = NLL_RAND_PNP.cpu().data.numpy() - NLL_BALD.cpu().data.numpy()
        BALD_Select_feature = np.sort(get_choice(observed_train_BALD - observed_train_BALD_old))[-10:]

        Results_list = ['%s' % (counter_loop * step),
                        '%.3f , %.3f , %.3f' % (
                            NLL_BALD.cpu().data.numpy(), NLL_RAND.cpu().data.numpy(),
                            NLL_RAND_PNP.data.cpu().numpy())] + [
                           NLL_diff_R, NLL_diff_P] + \
                       ['%.3f , %.3f' % (BALD_pattern_old, BALD_pattern_new)] + [
                           '%.3f , %.3f' % (RAND_pattern_old, RAND_pattern_new)] + [
                           '%.3f , %.3f' % (RAND_PNP_pattern_old, RAND_PNP_pattern_new)] + [
                           '%s' % (BALD_Select_feature)] + [String_BALD]+[String_RAND]  # + \
        # [get_choice(observed_train_BALD - observed_train_BALD_old).tolist()] \
        # + [get_choice(observed_train_BALD).tolist()] + [get_choice(observed_train_RAND).tolist()] + [
        #     get_choice(observed_train_RAND_PNP).tolist()]
        table = Table_format(table, Results_list)
        # Store the table
        # setting_num = 'Debug_4'
        store_Table(filepath_out + '/Setting%s_Seed_%s' % (setting_num, rand_seed) + '.txt',
                    table, title='Movielens_Setting_%s_Seed_%s' % (setting_num, rand_seed))

        print('Run:%s Data: %s' % (0, counter_loop * step))
        print('NLL_BALD:%s' % (NLL_BALD.cpu().data.numpy()))
        print('NLL_RAND:%s' % (NLL_RAND.cpu().data.numpy()))
        print('NLL_RAND_PNP:%s' % (NLL_RAND_PNP.cpu().data.numpy()))
        # Store the results
        RMSE_BALD_mat[runs, counter_loop], MAE_BALD_mat[runs, counter_loop], NLL_BALD_mat[runs, counter_loop], \
        RMSE_RAND_mat[runs, counter_loop], MAE_RAND_mat[
            runs, counter_loop], NLL_RAND_mat[runs, counter_loop], \
        RMSE_RAND_PNP_mat[runs, counter_loop], MAE_RAND_PNP_mat[runs, counter_loop], NLL_RAND_PNP_mat[
            runs, counter_loop] = RMSE_BALD.cpu().data.numpy(), MAE_BALD.cpu().data.numpy(), NLL_BALD.cpu().data.numpy(), \
                                  RMSE_RAND.cpu().data.numpy(), MAE_RAND.cpu().data.numpy(), NLL_RAND.cpu().data.numpy(), \
                                  RMSE_RAND_PNP.cpu().data.numpy(), MAE_RAND_PNP.cpu().data.numpy(), NLL_RAND_PNP.cpu().data.numpy()
        # Store Data
        if counter_selection % 20000==0:
            if Select_Split == True:
                filename_2 = 'Debug_Movielens_Step%s_total%s_CTTest_Conditional%s_%sBALD_%sBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_eps%s_Adam%s_MC_Opposite_StoredData_'\
                             % (step,max_selection,conditional_coef,BALD_coef,balance_coef,sigma_out,Drop_orig,args.step_sghmc,args.lr_sghmc)
            else:
                filename_2 = 'Debug_Movielens_Step%s_total%s_CTTest_Conditional%s_%sBALD_NoBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_eps%s_Adam%s_MC_Opposite_StoredData_' \
                             % ( step, max_selection, conditional_coef, BALD_coef,sigma_out, Drop_orig, args.step_sghmc, args.lr_sghmc)
            observed_train_BALD_rz_np = observed_train_BALD_rz.cpu().data.numpy()
            observed_train_RAND_rz_np = observed_train_RAND_rz.cpu().data.numpy()
            np.save(filepath_out + '/%sBALD_datanum_%s_Seed_%s.npy' % (filename_2,counter_loop * step,rand_seed), observed_train_BALD_rz_np)
            np.save(filepath_out + '/%sRAND_datanum_%s_Seed_%s.npy' % (filename_2,counter_loop * step,rand_seed), observed_train_RAND_rz_np)

        counter_loop+=1
        if idx_end>observed_train_BALD.shape[0]:
            idx_start=0
            idx_end=idx_start+batch_select
        else:
            idx_start=idx_end
            idx_end=idx_start+batch_select
        if Select_Split==True:
            filename_2 = 'Debug_Movielens_Step%s_total%s_CTTest_Conditional%s_%sBALD_%sBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_eps%s_Adam%s_MC_%sResults_' \
                     % ( step, max_selection, conditional_coef, BALD_coef, balance_coef,
                        sigma_out, Drop_orig, args.step_sghmc, args.lr_sghmc,suffix)
        else:
            filename_2 = 'Debug_Movielens_Step%s_total%s_CTTest_Conditional%s_%sBALD_NoBalance_%ssigma_0.25p0Temp_Drop%s_UpdateEncoder_eps%s_Adam%s_MC_%sResults_' \
                         % ( step, max_selection, conditional_coef, BALD_coef,
                            sigma_out, Drop_orig, args.step_sghmc, args.lr_sghmc,suffix)
        #Store the results
        BALD_Result = Results_Class(RMSE_test=RMSE_BALD_mat, MAE_test=MAE_BALD_mat, NLL_test=NLL_BALD_mat)

        RAND_Result = Results_Class(RMSE_test=RMSE_RAND_mat, MAE_test=MAE_RAND_mat, NLL_test=NLL_RAND_mat)
        RAND_PNP_Result = Results_Class(RMSE_test=RMSE_RAND_PNP_mat, MAE_test=MAE_RAND_PNP_mat,
                                        NLL_test=NLL_RAND_PNP_mat)


        filename = filepath_out
        save_data(filepath_out + '/%s'%(filename_2)+'BALD.pkl', BALD_Result)
        save_data(filepath_out + '/%s'%(filename_2)+'RAND.pkl', RAND_Result)
        save_data(filepath_out + '/%s'%(filename_2)+'RAND_PNP.pkl', RAND_PNP_Result)


