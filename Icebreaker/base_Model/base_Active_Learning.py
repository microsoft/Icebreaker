import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from Icebreaker.base_Model.base_Network import *
from Icebreaker.base_Model.base_BNN import *
from Icebreaker.base_Model.BNN_Network_zoo import *
from Util.Util_func import *
from Icebreaker.Dataloader.base_Dataloader import *
from torch.utils.data import DataLoader
from scipy.stats import bernoulli
from sklearn.utils import shuffle
class base_Active_Learning(object):
    def __init__(self,model,overall_data,active_scheme='1',rs=10,flag_clear_target_train=False,flag_clear_target_test=True,model_name='RAND'):
        # This is to initialize the active learning object, incorporating the training method and active learning module. The detailed training parameters are passed as dict argument in method.
        self.model=model # PNP, PNP_BNN ....
        self.overall_data=overall_data
        self.active_scheme=active_scheme
        self.random_seed=rs
        self.flag_clear_target_train=flag_clear_target_train
        self.flag_clear_target_test=flag_clear_target_test
        self.model_name=model_name
    def base_random_select(self,**kwargs):
        active_sample_number = kwargs['active_sample_number']
        test_input = kwargs['test_input']  # If this is the first time run, this should be all initialized at 0
        pool_data_tensor=kwargs['pool_data_tensor']
        total_user = pool_data_tensor.shape[0]
        # random select
        idx_array=[]
        for user_i in range(total_user):
            non_zero_idx=(torch.abs(pool_data_tensor[user_i,:])>0.).nonzero() # 1D idx tensor

            select_idx=non_zero_idx[torch.randperm(len(non_zero_idx))[0]] # random select 1 index
            test_input[user_i,select_idx]=pool_data_tensor[user_i,select_idx]
            # remove the pool
            pool_data_tensor[user_i,select_idx]=0.
            idx_array.append(select_idx)
        return test_input,idx_array,pool_data_tensor
    def base_active_learning_z_target_BNN(self,**kwargs):
        # EDDI Computation
        # This is to minimize the expected log likelihood of target variable
        active_sample_number=kwargs['active_sample_number']
        test_input=kwargs['test_input']
        pool_data_tensor=kwargs['pool_data_tensor']
        target_data_tensor=kwargs['target_data_tensor']
        test_input_mask=get_mask(test_input)

        sample_z_num=int(math.ceil(active_sample_number/20))
        sample_W_num=20

        # Sample X_phi and X_id
        z,_=self.model.sample_latent_variable(test_input,test_input_mask,size=sample_z_num) # size_z x N x latent

        if self.model.flag_LV:
            _, decode,_ = self.model.decoding(z, self.sigma_out, flag_record=False,
                                            size_W=sample_W_num)  # N_w x N_z x N x obs_dim
        else:
            _,decode=self.model.decoding(z,self.sigma_out,flag_record=False,size_W=sample_W_num) # N_w x N_z x N x obs_dim


        # Remove memory of decode and z
        decode = torch.tensor(decode.data)
        z = torch.tensor(z.data)
        # mask to select from target
        target_mask = get_mask(target_data_tensor)  # N x obs_dim

        target_mask_expand = torch.unsqueeze(torch.unsqueeze(target_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim

        target_candidate = target_mask_expand * decode  # N_w x N_z x N x obs_dim

        # mask to select from pool
        pool_mask=get_mask(pool_data_tensor)
        pool_mask_expand=torch.unsqueeze(torch.unsqueeze(pool_mask,dim=0),dim=0) # 1 x 1 x N x obs_dim
        pool_candidate = pool_mask_expand * decode  # N_w x N_z x N x obs_dim
        total_user = pool_candidate.shape[2]
        idx_array = []
        # iterate through each row
        for user_i in range(total_user):
            active_input_i_pool_target,active_input_i_target,non_zero_idx_dim=self.generate_active_learn_input_z_target_slice(test_input,pool_candidate,target_candidate,slice=user_i)
            active_input_i,non_zero_idx_dim=self.generate_active_learn_input_z_slice(test_input,pool_candidate,slice=user_i)

            # get mask
            mask_active_target=get_mask(active_input_i_target) # tot x obs_dim
            mask_active_pool_target=get_mask(active_input_i_pool_target) # tot x d x obs_dim
            mask_active=get_mask(active_input_i)
            # Encoding
            encoding_target=self.model._encoding(active_input_i_target, mask_active_target) # total x 2*latent
            encoding_pool_target=self.model._encoding(active_input_i_pool_target,mask_active_pool_target) # tot x d x 2*latent
            encoding = self.model._encoding(active_input_i, mask_active)  # total x d x 2*latent
            # clear memory
            encoding_target=torch.tensor(encoding_target.data)
            encoding_pool_target=torch.tensor(encoding_pool_target.data)
            encoding=torch.tensor(encoding.data)
            # Compute the KL term
            encoding_target=torch.unsqueeze(encoding_target,dim=1).repeat(1,encoding_pool_target.shape[1],1)# tot x d x 2*latent
            KL=self._gaussian_KL(encoding_pool_target,encoding_target) # tot x d
            # Compute the entropy
            entropy = self._gaussian_encoding_entropy(encoding)  # total x d

            loss=-entropy-KL # tot x d
            mean_loss=torch.mean(loss,dim=0) # d
            # select the maximum value
            _, idx_max = torch.max(mean_loss, dim=0)

            # Sample according to probablility
            # mean_scale = torch.mean(torch.abs(mean_loss))  # 1
            # temp = (0.2 + 0) * mean_scale
            # mean_loss_prob = torch.clamp(
            #     F.softmax(mean_loss / temp), max=1.)  # d

            #idx_prob = torch.multinomial(mean_loss_prob, 1) # 1

            # original index
            idx_select = non_zero_idx_dim[idx_max]
            #idx_select = non_zero_idx_dim[idx_prob]

            idx_array.append(idx_select)
            # update test_input
            test_input[user_i, idx_select] = pool_data_tensor[user_i, idx_select]

            # update the pool by removing the selected ones
            pool_data_tensor[user_i, idx_select] = 0.
        return test_input, idx_array, pool_data_tensor

    def _gaussian_KL(self,encoding1,encoding2):
        # assert if the shape matches, normally is tot x d x 2*latent
        assert encoding1.shape==encoding2.shape, ' Inconsistent encoding shapes'
        mean1,mean2=encoding1[:,:,0:self.model.latent_dim],encoding2[:,:,0:self.model.latent_dim] # tot x d x latent
        if self.model.flag_log_q==True:
            sigma1, sigma2 = torch.sqrt(torch.exp(encoding1[:, :, self.model.latent_dim:])), torch.sqrt(torch.exp(
                encoding2[:, :, self.model.latent_dim:]))  # tot x d x latent
        else:
            sigma1,sigma2=torch.sqrt(encoding1[:,:,self.model.latent_dim:]**2),torch.sqrt(encoding2[:,:,self.model.latent_dim:]**2) # tot x d x latent
        KL=torch.log(sigma2/sigma1)+(sigma1**2+(mean1-mean2)**2)/(2*sigma2**2)-0.5 # tot x d x latent
        KL=torch.sum(KL,dim=2) # tot x d
        return KL


    def _gaussian_encoding_entropy(self,encoding):
        mean=encoding[:,:,0:self.model.latent_dim] # total x d x latent
        if self.model.flag_log_q==True:
            sigma=torch.sqrt(torch.exp((encoding[:,:, self.model.latent_dim:])))
        else:
            sigma = torch.sqrt((encoding[:,:, self.model.latent_dim:]) ** 2)
        entropy=0.5*torch.log(2*3.1415926*2.71828*(sigma**2)) # total x d x latent
        entropy=torch.sum(entropy,dim=2) # total x d
        return entropy





    def generate_active_learn_input_z_slice(self,test_input,pool_candidate,slice):
        # Generate the input for computing the BALD, slice indicate the user row number for test_input
        pool_candidate_slice=pool_candidate.view(-1,pool_candidate.shape[2],pool_candidate.shape[3]) # (N_zxN_W) x N x obs_dim
        pool_candidate_slice=torch.unsqueeze(pool_candidate_slice[:,slice,:],dim=2) # (total_sample_num) x obs_dim x 1
        non_zero_idx=(torch.abs(pool_candidate_slice)>0.).nonzero() # 3D tensor
        non_zero_idx_dim=non_zero_idx[:,1]
        non_zero_idx_dim=torch.unique(non_zero_idx_dim) #1D Tensor
        total_pool_size=non_zero_idx_dim.shape[0]
        total_sample_number=pool_candidate_slice.shape[0]
        # Non zero pool
        pool_candidate_slice=torch.index_select(pool_candidate_slice,dim=1,index=non_zero_idx_dim) # (total) x d_pool x 1
        # index array
        non_zero_idx_array=torch.unsqueeze(torch.unsqueeze(non_zero_idx_dim,dim=0).repeat(total_sample_number,1),dim=2) # N x d_pool x 1
        # replicate the test_input
        test_input_slice=test_input[slice,:] # obs_dim
        test_input_slice=torch.unsqueeze(torch.unsqueeze(test_input_slice,dim=0),dim=0).repeat(total_sample_number,total_pool_size,1)

        active_input=test_input_slice.scatter_(dim=2,index=non_zero_idx_array,src=pool_candidate_slice) # total x d x obs_dim

        return active_input,non_zero_idx_dim


    def generate_active_learn_input_z_target_slice(self,test_input,pool_candidate,target_candidate,slice):
        # Generate X_target,X_o
        target_candidate_slice=target_candidate.view(-1,target_candidate.shape[2],target_candidate.shape[3]) # (tot) x N x obs_dim
        target_candidate_slice=target_candidate_slice[:,slice,:] # tot x obs_dim
        non_zero_idx_target = (torch.abs(target_candidate_slice) > 0.).nonzero() # 2D tensor
        non_zero_idx_dim_target = non_zero_idx_target[:, 1]
        non_zero_idx_dim_target = torch.unique(non_zero_idx_dim_target) # 1D Tensor
        total_sample_number = target_candidate_slice.shape[0]
        # Non zero target
        target_candidate_slice = torch.index_select(target_candidate_slice, dim=1,
                                                  index=non_zero_idx_dim_target)  # (total) x d_target
        # Index array
        non_zero_idx_array_target = torch.unsqueeze(non_zero_idx_dim_target, dim=0).repeat(total_sample_number, 1)# tot x d_pool
        # replicate the test_input
        test_input_slice = test_input[slice, :]  # obs_dim
        test_input_slice = torch.unsqueeze(test_input_slice, dim=0).repeat(total_sample_number, 1) # tot x obs_dim
        # X_target,X_o
        active_input_target = test_input_slice.scatter_(dim=1, index=non_zero_idx_array_target,
                                                 src=target_candidate_slice)  # total x obs_dim
        # Generate X_target,X_i,X_o
        pool_candidate_slice = pool_candidate.view(-1, pool_candidate.shape[2],
                                                   pool_candidate.shape[3])  # (N_zxN_W) x N x obs_dim
        pool_candidate_slice = torch.unsqueeze(pool_candidate_slice[:, slice, :],
                                               dim=2)  # (total_sample_num) x obs_dim x 1
        non_zero_idx = (torch.abs(pool_candidate_slice) > 0.).nonzero()  # 3D tensor
        non_zero_idx_dim = non_zero_idx[:, 1]
        non_zero_idx_dim = torch.unique(non_zero_idx_dim)  # 1D Tensor
        total_pool_size = non_zero_idx_dim.shape[0]
        total_sample_number = pool_candidate_slice.shape[0]
        # Non zero pool
        pool_candidate_slice = torch.index_select(pool_candidate_slice, dim=1,
                                                  index=non_zero_idx_dim)  # (total) x d_pool x 1
        # index array
        non_zero_idx_array = torch.unsqueeze(torch.unsqueeze(non_zero_idx_dim, dim=0).repeat(total_sample_number, 1),
                                             dim=2)  # tot x d_pool x 1

        # replicate the active_input_target
        active_input_pool_target=torch.unsqueeze(active_input_target,dim=1).repeat(1,total_pool_size,1) # tot x d x obs_dim


        active_input_pool_target=active_input_pool_target.scatter_(dim=2,index=non_zero_idx_array,src=pool_candidate_slice) # total x d x
        return active_input_pool_target,active_input_target,non_zero_idx_dim

class base_Active_Learning_Decoder(base_Active_Learning):
    def __init__(self,Adam_encoder,Adam_decoder,Adam_embedding,sigma_out,Optim_settings,*args,**kwargs):
        super(base_Active_Learning_Decoder,self).__init__(*args,**kwargs)
        self.Adam_encoder = Adam_encoder
        self.Adam_decoder = Adam_decoder
        self.Adam_embedding = Adam_embedding
        self.sigma_out=sigma_out
        self.Optim_settings=Optim_settings


    def _get_pretrain_data(self,target_dim=-1,**kwargs):
        raise NotImplementedError
    def _data_preprocess(self,target_dim=-1,**kwargs):
        raise NotImplementedError
    def base_active_learning_decoder(self,coef=0.999,eps=0.4,balance_prop=0.25,temp_scale=0.25,flag_initial=False,Select_Split=True,**kwargs):
        raise NotImplementedError

    def get_target_variable(self,observed_train,observed_train_before,target_dim,train_data=None):
        raise NotImplementedError

    def base_random_select_training(self,balance_prop=0.25,Select_Split=True,flag_initial=False,**kwargs):
        observed_train = kwargs['observed_train']
        pool_data = kwargs['pool_data']
        pool_data_cp=torch.tensor(pool_data.data)
        step = kwargs['step']  # This is to define how many data points to select for each active learning
        obs=observed_train.shape[1]
        selection_scheme = kwargs['selection_scheme']
        if selection_scheme == 'overall':
            print('%s selection scheme with Select_Split %s' % (selection_scheme, Select_Split))
        else:
            raise NotImplementedError
        if selection_scheme=='overall':
            if not flag_initial:
                if Select_Split==True:
                    print('%s Explore'%(self.model_name))

                    step_new=int((1-balance_prop)*step)
                    #step_old=step-step_new
                    #Select from new
                    pool_data_cp_un=assign_zero_row_2D_with_target_reverse(pool_data_cp,observed_train)
                    idx_un,num_selected_un,flag_full_un=Random_select_idx(pool_data_cp_un,obs,step_new)
                    if flag_full_un:
                        print('%s No more new user'%(self.model_name))
                        step_old=step
                    else:
                        step_old=step-num_selected_un
                    # Select from old
                    pool_data_cp_ob = assign_zero_row_2D_with_target(pool_data_cp, observed_train)
                    idx_ob,num_selected_ob,flag_full=Random_select_idx(pool_data_cp_ob,obs,step_old)

                    if not flag_full_un:
                        idx=torch.cat((idx_ob,idx_un),dim=0)
                    else:
                        idx=idx_ob
                    num_selected=num_selected_ob+num_selected_un
                else:
                    idx, num_selected, flag_full = Random_select_idx(pool_data_cp, obs, step)
            else:

                idx,num_selected,flag_full=Random_select_idx(pool_data_cp,obs,step)

            if not flag_full:
                observed_train,pool_data=self._apply_selected_idx(observed_train, pool_data, idx)
        else:
            raise NotImplementedError

        return observed_train,pool_data,flag_full,num_selected


    def train_BNN(self,flag_pretrain=False,flag_stream=False,flag_hybrid=False,target_dim=-1,conditional_coef=0.75,flag_results=False,flag_imputation=False,**kwargs):
        # Train the model using train data, arguments are passed through **kwargs, random_seed are set outside, most basic one, not applicable to streaming version.


        epoch_max = kwargs['epoch']

        flag_dataloader = kwargs['flag_dataloader']


        batch_size = kwargs['batch_size']
        z_sigma_prior = kwargs['z_sigma_prior']
        W_sigma_prior = kwargs['W_sigma_prior']
        valid_data=kwargs['valid_data']
        sigma_out = self.sigma_out




        Drop_p = kwargs['Drop_p']
        # Train data
        train_data=kwargs['observed_train']
        coef_KL_coef_W=1. #num_points/num_pool
        # if epoch<50:
        #     epoch=50
        if flag_pretrain==False:

            epoch=epoch_max#int(np.minimum((epoch_max-50)/(25)*counter_loop,500))
        else:
            coef_KL_coef_W = 1.
            epoch=epoch_max

        # Delete all zero rows
        train_data_size=get_train_data_size(train_data)
        # Modify the batch size
        if batch_size>train_data_size:
            batch_size=int(train_data_size/2)



        # Optimizer
        flag_reset_optim=kwargs['flag_reset_optim']

        # Optimizer are defined based on whether to reset the optim
        if flag_reset_optim==False:
            # Use the optimizer inside the class
            Adam_encoder = self.Adam_encoder
            Adam_decoder = self.Adam_decoder
            Adam_embedding = self.Adam_embedding

        else:
            # replicate the Optimizer outside the train loop
            Adam_encoder  = torch.optim.Adam(
                list(self.model.encoder_before_agg.parameters()) + list(self.model.encoder_after_agg.parameters()), lr=self.Optim_settings['lr'],
                betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                weight_decay=self.Optim_settings['weight_decay'])
            Adam_decoder = torch.optim.Adam(
                list(self.model.decoder.parameters()), lr=self.Optim_settings['lr'],
                betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                weight_decay=0)
            Adam_embedding = torch.optim.Adam(
                [self.model.encode_embedding, self.model.encode_bias], lr=self.Optim_settings['lr'],
                betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                weight_decay=self.Optim_settings['weight_decay'])


        flag_BNN = kwargs['flag_BNN']
        KL_coef_W = kwargs['KL_coef_W']
        if flag_results==True:
            NLL_mat,MAE_mat,RMSE_mat=np.zeros(epoch),np.zeros(epoch),np.zeros(epoch)
        # Whether to use dataloader
        if flag_dataloader:
            train_dataset = base_UCI_Dataset(train_data, transform=None, flag_GPU=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # Training the model
            valid_NLL_list = collections.deque(maxlen=10)
            mean_Acc_valid_min = 10.
            counter_valid_break = 0
            acc_ELBO_feature = 0.
            for ep in range(epoch):

                if (ep + 1) % 100 == 0:
                    print('PNP_BNN Epoch:%s/%s' % (ep + 1, epoch))
                for idx, data in enumerate(train_loader):
                    # zero grad
                    Adam_encoder.zero_grad()
                    Adam_decoder.zero_grad()
                    Adam_embedding.zero_grad()
                    # Generate drop mask
                    mask = get_mask(data)
                    # Debug variable artificial missingness
                    Drop_p_var = np.minimum(np.random.rand(1), Drop_p)
                    #Drop_p_var=Drop_p
                    #
                    mask_drop = np.array([bernoulli.rvs(1 - Drop_p_var, size=data.shape[1])] * data.shape[0])
                    mask_drop = torch.from_numpy(mask_drop).float().cuda()
                    if flag_hybrid:
                        mask_drop_hybrid=torch.tensor(mask_drop.data)
                        mask_drop_hybrid[:,target_dim]=1. # Reserve the target dim
                        mask_drop_Xdy=torch.tensor(mask_drop.data)
                        mask_drop_Xdy[:,-1]=0.
                        mask_target_ELBO = mask * mask_drop_hybrid
                        mask=mask*mask_drop
                    else:
                        mask=mask*mask_drop#least_one_left_mask(mask,mask_drop)#mask*mask_drop
                    ### DEBUG ###
                    # mask[:,-1]=1
                    #############

                    if flag_stream==False:
                        elbo_mean, elbo_mean_feature = self.model.ELBO(data, mask, W_sigma_prior, z_sigma_prior, sigma_out,
                                                                   epoch=ep, KL_coef_W=KL_coef_W,
                                                                   train_data_size=train_data_size,
                                                                   flag_BNN=flag_BNN,flag_stream=flag_stream,coef_KL_coef_W=coef_KL_coef_W)
                    else:
                        pre_W_mean, pre_b_mean, pre_W_sigma, pre_b_sigma = kwargs['pre_W_mean'], kwargs['pre_b_mean'], kwargs['pre_W_sigma'], kwargs['pre_b_sigma']
                        elbo_mean, elbo_mean_feature = self.model.ELBO(data, mask, W_sigma_prior, z_sigma_prior,
                                                                       sigma_out,
                                                                       epoch=ep, KL_coef_W=KL_coef_W,
                                                                       train_data_size=train_data_size,coef_KL_coef_W=coef_KL_coef_W,
                                                                       flag_BNN=flag_BNN,flag_stream=flag_stream,pre_W_sigma=pre_W_sigma,pre_b_sigma=pre_b_sigma,pre_W_mean=pre_W_mean,pre_b_mean=pre_b_mean)

                    if flag_hybrid: # Enable conditional training
                        if flag_stream==False:
                            elbo_mean_target, elbo_mean_feature_target = self.model.target_ELBO(data, mask_target_ELBO, W_sigma_prior,
                                                                           sigma_out,
                                                                           epoch=ep, KL_coef_W=KL_coef_W,
                                                                           train_data_size=train_data_size,
                                                                           flag_BNN=flag_BNN, flag_stream=flag_stream,
                                                                           coef_KL_coef_W=coef_KL_coef_W,target_dim=target_dim)
                        else:
                            if self.model.flag_LV:
                                raise NotImplementedError
                            pre_W_mean, pre_b_mean, pre_W_sigma, pre_b_sigma = kwargs['pre_W_mean'], kwargs[
                                'pre_b_mean'], kwargs['pre_W_sigma'], kwargs['pre_b_sigma']
                            elbo_mean_target, elbo_mean_feature_target = self.model.target_ELBO(data, mask_target_ELBO, W_sigma_prior,
                                                                           sigma_out,
                                                                           epoch=ep, KL_coef_W=KL_coef_W,
                                                                           train_data_size=train_data_size,
                                                                           coef_KL_coef_W=coef_KL_coef_W,
                                                                           flag_BNN=flag_BNN, flag_stream=flag_stream,target_dim=target_dim,
                                                                           pre_W_sigma=pre_W_sigma,
                                                                           pre_b_sigma=pre_b_sigma,
                                                                           pre_W_mean=pre_W_mean, pre_b_mean=pre_b_mean)

                        elbo_mean=conditional_coef*elbo_mean+(1.-conditional_coef)*elbo_mean_target
                        elbo_feature_target=conditional_coef*elbo_mean_feature+(1.-conditional_coef)*elbo_mean_feature_target

                    acc_ELBO_feature=acc_ELBO_feature+elbo_mean_feature.data.cpu().numpy()
                    loss = -elbo_mean
                    loss.backward()
                    # Update Parameters
                    Adam_encoder.step()
                    Adam_decoder.step()
                    Adam_embedding.step()
                if (ep+1)%100==0:
                    mask_train=get_mask(train_data)
                    ##TODO: Disable this if movielens
                    if not flag_imputation:
                        train_NLL,_=self.model.test_log_likelihood(train_data, train_data.clone().detach(), mask_train, sigma_out, size=None)

                        print('Training loss:%s'%(train_NLL.cpu().data.numpy()))
                    else:
                        print('Training NLL:%s'%('Not Applicable'))

                    acc_ELBO_feature=0.

                # Evaluate the validation NLL
                if (ep + 1) % 10 == 0 and valid_data is not None:
                    raise NotImplementedError('Early stop has not been implemented yet')

                if (ep+1)%100==0:
                    RMSE_test, MAE_test, NLL_test = Test_UCI_batch(self.model, self.test_input_tensor,
                                                                   self.test_target_tensor,
                                                                   sigma_out_scale=sigma_out,
                                                                   split=10, flag_model='PNP_BNN')
                    print('NLL:%s RMSE:%s' % (
                        NLL_test.cpu().data.numpy(),RMSE_test.cpu().data.numpy()))

                if flag_results==True:
                    RMSE_test, MAE_test, NLL_test = Test_UCI_batch(self.model, self.test_input_tensor,
                                                                   self.test_target_tensor,
                                                                   sigma_out_scale=sigma_out,
                                                                   split=10, flag_model='PNP_BNN')
                    RMSE_mat[ep], MAE_mat[ep], NLL_mat[
                        ep] = RMSE_test.cpu().data.numpy(), MAE_test.cpu().data.numpy(), NLL_test.cpu().data.numpy()

        else:
            raise NotImplementedError
        if flag_results==True:
            return RMSE_mat,MAE_mat,NLL_mat


    def _apply_selected_idx(self,observed_train,pool_data,idx):
        for i in range(idx.shape[0]):
            row=idx[i,0]
            column=idx[i,1]
            #Assign to observed train
            observed_train[row,column]=pool_data[row,column]
            # Remove pool_data
            pool_data[row,column]=0.
        return observed_train,pool_data