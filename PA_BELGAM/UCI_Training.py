import sys
import os
cwd=os.getcwd()
cwd_parent=os.path.abspath('..')
sys.path.append(cwd)
sys.path.append(cwd_parent)
sys.path.append(cwd_parent+'/Util')
sys.path.append(cwd_parent+'/Icebreaker') #for data loader
import torch
from PA_BELGAM.PA_BELGAM_Dataloader import *
from PA_BELGAM.PA_BELGAM_model import *
from PA_BELGAM.PA_BELGAM_infer import *
import os
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
rand_seed_list=[20,30,40,50,60,70,80,90,100,110]
ratio_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]

# rand_seed_list=[20,30,40,50,60,70,80,90,100,110]
# ratio_list=[0.1,0.2,0.3,]

rand_len=len(rand_seed_list)
ratio_len=len(ratio_list)
for ratio_idx in range(ratio_len):
    for runs in range(rand_len):
        # fix random seed
        rand_seed=rand_seed_list[runs]
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        # Load UCI data
        root_dir=cwd_parent+'/Icebreaker/Dataloader/data/uci'
        Data_mat=load_UCI_data('0',root_dir,flag_normalize=True,min_data=0.,normalization='True_Normal',flag_shuffle=True)
        # split train and test
        train_data,test_data=UCI_preprocessing(Data_mat,split_ratio=0.8,reduce_train=ratio_list[ratio_idx])

        # Model Settings:
        latent_dim=5
        output_dim=1
        input_dim=train_data.shape[1]-output_dim
        ## Network architecture
        prior_net_layer_num=1
        prior_net_hidden=[20]
        recog_net_layer_num=1
        recog_net_hidden=[20]
        decoder_layer_num=1
        decoder_hidden=[20]
        ## Output settings
        output_const=1.
        add_const=0.
        sample_Z=50
        KL_coef=1.
        hybrid_coef=0.5
        flag_log_q=True
        sigma_out=0.15
        # Define Model

        PA_VAE=PA_BELGAM(latent_dim,input_dim,output_dim,prior_net_layer_num=prior_net_layer_num,prior_net_hidden=prior_net_hidden,
                         recog_net_layer_num=recog_net_layer_num,recog_net_hidden=recog_net_hidden,decoder_layer_num=decoder_layer_num,
                         decoder_hidden=decoder_hidden,output_const=output_const,add_const=add_const,sample_Z=sample_Z,KL_coef=KL_coef,
                         hybrid_coef=hybrid_coef,flag_log_q=flag_log_q
                         )
        Adam_VAE=PA_BELGAM(latent_dim,input_dim,output_dim,prior_net_layer_num=prior_net_layer_num,prior_net_hidden=prior_net_hidden,
                         recog_net_layer_num=recog_net_layer_num,recog_net_hidden=recog_net_hidden,decoder_layer_num=decoder_layer_num,
                         decoder_hidden=decoder_hidden,output_const=output_const,add_const=add_const,sample_Z=sample_Z,KL_coef=KL_coef,
                         hybrid_coef=hybrid_coef,flag_log_q=flag_log_q
                         )

        # Define inference
        PA_SGHMC=SGHMC(PA_VAE)
        PA_ADAM=SGHMC(Adam_VAE)
        #Define Optimizer and list_p_z
        lr_adam=0.002

        Adam_encoder=torch.optim.Adam(list(PA_VAE.prior_net.parameters())+list(PA_VAE.recog_net.parameters()),lr=lr_adam,betas=(0.9,0.99))
        list_p_z=list(PA_VAE.prior_net.parameters())+list(PA_VAE.recog_net.parameters())

        # Training settings
        lr_sghmc=5e-5


        PA_RMSE_comp,PA_MAE_comp,PA_NLL_comp=PA_SGHMC.train_SGHMC(train_data,eps=lr_sghmc,max_sample_size=80,tot_epoch=2000,thinning=5,result_interval=2,flag_results=True,
                             Adam_encoder=Adam_encoder,list_p_z=list_p_z,test_data=test_data,sigma_out=sigma_out,flag_adam=False
                             )


        # Adam optimizer for decoder
        Adam_encoder_adam=torch.optim.Adam(list(Adam_VAE.prior_net.parameters())+list(Adam_VAE.recog_net.parameters()),lr=lr_adam,betas=(0.9,0.99))
        list_p_z_adam=list(Adam_VAE.prior_net.parameters())+list(Adam_VAE.recog_net.parameters())
        ADAM_RMSE_comp,ADAM_MAE_comp,ADAM_NLL_comp=PA_ADAM.train_SGHMC(train_data,eps=lr_adam,max_sample_size=80,tot_epoch=2000,thinning=5,result_interval=2,flag_results=True,Adam_encoder=Adam_encoder_adam,list_p_z=list_p_z_adam,
                            test_data=test_data,sigma_out=sigma_out,flag_adam=True
                            )
        # Process the results
        if runs==0:
            PA_RMSE,PA_MAE,PA_NLL=np.expand_dims(PA_RMSE_comp,axis=0),np.expand_dims(PA_MAE_comp,axis=0),np.expand_dims(PA_NLL_comp,axis=0)
            ADAM_RMSE, ADAM_MAE, ADAM_NLL = np.expand_dims(ADAM_RMSE_comp, axis=0), np.expand_dims(ADAM_MAE_comp,
                                                                                          axis=0), np.expand_dims(
                ADAM_NLL_comp, axis=0)
        else:
            PA_RMSE_comp, PA_MAE_comp, PA_NLL_comp = np.expand_dims(PA_RMSE_comp, axis=0), np.expand_dims(PA_MAE_comp,
                                                                                          axis=0), np.expand_dims(
                PA_NLL_comp, axis=0)
            PA_RMSE, PA_MAE, PA_NLL=np.concatenate((PA_RMSE,PA_RMSE_comp),axis=0),np.concatenate((PA_MAE,PA_MAE_comp),axis=0),np.concatenate((PA_NLL,PA_NLL_comp),axis=0)

            ADAM_RMSE_comp, ADAM_MAE_comp, ADAM_NLL_comp = np.expand_dims(ADAM_RMSE_comp, axis=0), np.expand_dims(ADAM_MAE_comp,
                                                                                                         axis=0), np.expand_dims(
                ADAM_NLL_comp, axis=0)
            ADAM_RMSE, ADAM_MAE, ADAM_NLL = np.concatenate((ADAM_RMSE, ADAM_RMSE_comp), axis=0), np.concatenate((ADAM_MAE, ADAM_MAE_comp),
                                                                                                      axis=0), np.concatenate(
                (ADAM_NLL, ADAM_NLL_comp), axis=0)
    if ratio_idx==0:
        PA_RMSE_ratio, PA_MAE_ratio, PA_NLL_ratio = np.expand_dims(PA_RMSE, axis=0), np.expand_dims(PA_MAE,
                                                                                       axis=0), np.expand_dims(
            PA_NLL, axis=0)
        ADAM_RMSE_ratio, ADAM_MAE_ratio, ADAM_NLL_ratio = np.expand_dims(ADAM_RMSE, axis=0), np.expand_dims(ADAM_MAE,
                                                                                               axis=0), np.expand_dims(
            ADAM_NLL, axis=0)
    else:
        PA_RMSE_comp, PA_MAE_comp, PA_NLL_comp = np.expand_dims(PA_RMSE, axis=0), np.expand_dims(PA_MAE,
                                                                                                      axis=0), np.expand_dims(
            PA_NLL, axis=0)
        PA_RMSE_ratio, PA_MAE_ratio, PA_NLL_ratio = np.concatenate((PA_RMSE_ratio, PA_RMSE_comp), axis=0), np.concatenate((PA_MAE_ratio, PA_MAE_comp),
                                                                                                  axis=0), np.concatenate(
            (PA_NLL_ratio, PA_NLL_comp), axis=0)

        ADAM_RMSE_comp, ADAM_MAE_comp, ADAM_NLL_comp = np.expand_dims(ADAM_RMSE, axis=0), np.expand_dims(
            ADAM_MAE,
            axis=0), np.expand_dims(
            ADAM_NLL, axis=0)
        ADAM_RMSE_ratio, ADAM_MAE_ratio, ADAM_NLL_ratio = np.concatenate((ADAM_RMSE_ratio, ADAM_RMSE_comp), axis=0), np.concatenate(
            (ADAM_MAE_ratio, ADAM_MAE_comp),
            axis=0), np.concatenate(
            (ADAM_NLL_ratio, ADAM_NLL_comp), axis=0)

# save the results
Result_path=cwd+'/Results/UCI0/'

np.save(Result_path+'PA_RMSE.npy',PA_RMSE_ratio)
np.save(Result_path+'PA_MAE.npy',PA_MAE_ratio)
np.save(Result_path+'PA_NLL.npy',PA_NLL_ratio)
np.save(Result_path+'ADAM_RMSE.npy',ADAM_RMSE_ratio)
np.save(Result_path+'ADAM_MAE.npy',ADAM_MAE_ratio)
np.save(Result_path+'ADAM_NLL.npy',ADAM_NLL_ratio)








