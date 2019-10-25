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
from base_Model.base_Infer import *

# This file mainly used for defining relevant active learning functions for Icebreaker

class base_Active_Learning_SGHMC(object):
    def __init__(self,model,Infer_model,overall_data,sigma_out,Optim_settings,Adam_encoder,Adam_embedding,flag_clear_target_train,flag_clear_target_test,rs=10,model_name='SGHMC Active'):
        self.model=model
        self.Infer_model=Infer_model
        self.overall_data=overall_data
        self.random_seed=rs
        self.flag_clear_target_train=flag_clear_target_train
        self.flag_clear_target_test=flag_clear_target_test
        self.model_name=model_name
        self.sigma_out=sigma_out
        self.Optim_settings=Optim_settings
        self.Adam_encoder=Adam_encoder
        self.Adam_embedding=Adam_embedding
    def _get_pretrain_data(self,target_dim=-1,**kwargs):
        rs=self.random_seed
        pretrain_data_number=kwargs['pretrain_number']
        # Shuffle the train data
        self.train_data=shuffle(self.train_data,random_state=rs)
        _, pretrain_data, _, _ = train_test_split(self.train_data, self.train_data, test_size=pretrain_data_number,
                                                       random_state=rs, shuffle=False)

        self.train_data[self.train_data.shape[0]-pretrain_data.shape[0]:,:]=0.
        self.init_observed_train=np.zeros(self.train_data.shape) # Original size
        self.init_observed_train[self.train_data.shape[0]-pretrain_data.shape[0]:,:]=pretrain_data


        self.train_data_tensor=torch.from_numpy(self.train_data).float().cuda()
        self.pretrain_data_tensor=torch.from_numpy(pretrain_data).float().cuda()
        self.init_observed_train_tensor=torch.from_numpy(self.init_observed_train).float().cuda()
        self.train_pool_tensor=self.train_data_tensor.clone().detach()
        if self.flag_clear_target_train:
            self.train_pool_tensor[:,target_dim]=0. # Remove the target dim
    def _data_preprocess(self,target_dim=-1,test_missing=0.2,**kwargs):
        rs = self.random_seed
        test_size = kwargs['test_size']
        missing_prop = kwargs['missing_prop']

        # Split user first, then mask by certain proportion aligned with train data. Train data are further split by proportion for mimicing variable size training dataset.
        train_data, test_data, _, _ = train_test_split(self.overall_data, self.overall_data, test_size=test_size,
                                                       random_state=rs, shuffle=True)
        # Split to get validation dataset
        train_data,valid_data,_,_ = train_test_split(train_data,train_data,test_size=0.1,random_state=rs,shuffle=True)

        # Mask the train data
        train_mask, train_mask_other = base_generate_mask_incomplete(train_data, mask_prop=missing_prop+1e-5)
        valid_mask,valid_mask_other =base_generate_mask_incomplete(valid_data,mask_prop=missing_prop+1e-5)

        if self.flag_clear_target_train==True: # Reserve the target variable
            train_mask[:,target_dim]=1.
            train_mask_other[:,target_dim]=0.
            valid_mask[:,target_dim]=1.
            valid_mask_other[:,target_dim]=0.
            valid_mask_input=np.ones(valid_mask.shape)
            valid_mask_input[:,target_dim]=0.
            valid_mask_target=np.zeros(valid_mask.shape)
            valid_mask_target[:,target_dim]=1.
        else:
            valid_mask_input=copy.deepcopy(valid_mask)
            valid_mask_target=copy.deepcopy(valid_mask_other)



        self.train_data = train_data * train_mask
        self.train_other = train_data * train_mask_other
        self.valid_data=valid_data*valid_mask
        self.valid_other=valid_data*valid_mask_other

        if self.flag_clear_target_test==True:
            test_mask = np.ones(test_data.shape)
            test_mask[:, target_dim] = 0
            target_mask = np.zeros(test_data.shape)
            target_mask[:, target_dim] = 1
        else:
            # NO CLEAR TARGET
            test_mask, target_mask = base_generate_mask_incomplete(test_data, mask_prop=missing_prop+test_missing)
            _,valid_mask_target=base_generate_mask_incomplete(valid_data, mask_prop=missing_prop+test_missing)
        ####### Debug #########
        self.train_data=self.train_data
        # self.train_data[:,1:self.train_data.shape[1]-1]=0.
        # test_mask[:,1:self.train_data.shape[1]-1]=0.

        #######################
        # transform to tensor
        self.valid_data_input=valid_data*valid_mask_input
        self.valid_data_target = valid_data * valid_mask_target
        self.valid_data_tensor=torch.from_numpy(self.valid_data).float().cuda()
        self.valid_data_target_tensor=torch.from_numpy(self.valid_data_target).float().cuda()
        self.valid_data_input_tensor = torch.from_numpy(self.valid_data_input).float().cuda()

        self.test_input = test_data * test_mask
        self.test_target = test_data * target_mask
        self.test_target_tensor = torch.from_numpy(self.test_target).float().cuda()
        self.test_input_tensor = torch.from_numpy(self.test_input).float().cuda()
        self.train_data_tensor=torch.from_numpy(self.train_data).float().cuda()
        self.init_observed_train=np.zeros(self.train_data_tensor.shape)
        self.init_observed_train_tensor=torch.from_numpy(self.init_observed_train).float().cuda()
        self.train_pool_tensor=torch.tensor(self.train_data_tensor.data)
        if self.flag_clear_target_train==True:
            self.train_pool_tensor[:,target_dim]=0. # Remove the target variable

    def _save_data(self,filepathname):
        # Save train data
        np.save(filepathname+'train_data.npy',self.train_data)
        # Test data
        np.save(filepathname+'test_input_data.npy',self.test_input)
        np.save(filepathname+'test_target_data.npy',self.test_target)
        # Store valid data
        np.save(filepathname+'valid_data.npy',self.valid_data)
        np.save(filepathname+'valid_data_input.npy',self.valid_data_input)
        np.save(filepathname+'valid_data_target.npy',self.valid_data_target)
    def _load_data(self,filepathname,target_dim=-1):
        self.train_data=np.load(filepathname+'train_data.npy')
        self.train_data_tensor = torch.from_numpy(self.train_data).float().cuda()
        self.test_input=np.load(filepathname+'test_input_data.npy')
        self.test_input_tensor = torch.from_numpy(self.test_input).float().cuda()
        self.test_target=np.load(filepathname+'test_target_data.npy')
        self.test_target_tensor = torch.from_numpy(self.test_target).float().cuda()

        self.valid_data=np.load(filepathname+'valid_data.npy')
        self.valid_data_tensor=torch.from_numpy(self.valid_data).float().cuda()

        self.valid_data_input=np.load(filepathname+'valid_data_input.npy')
        self.valid_data_input_tensor = torch.from_numpy(self.valid_data_input).float().cuda()

        self.valid_data_target=np.load(filepathname+'valid_data_target.npy')
        self.valid_data_target_tensor=torch.from_numpy(self.valid_data_target).float().cuda()

        self.init_observed_train = np.zeros(self.train_data_tensor.shape)
        self.init_observed_train_tensor = torch.from_numpy(self.init_observed_train).float().cuda()
        self.train_pool_tensor = torch.tensor(self.train_data_tensor.data)
        if self.flag_clear_target_train == True:
            self.train_pool_tensor[:, target_dim] = 0.  # Remove the target variable



    def train_BNN(self,observed_train,eps=0.01,max_sample_size=20,tot_epoch=100,thinning=50,hyper_param_update=200,sample_int=1,flag_hybrid=False,flag_reset_optim=True,flag_imputation=False,W_dict_init=None,**kwargs):

        if flag_reset_optim==True:
            # Redefine the optimizer
            Adam_encoder = torch.optim.Adam(
                list(self.model.encoder_before_agg.parameters()) + list(self.model.encoder_after_agg.parameters()),
                lr=self.Optim_settings['lr_sghmc'],
                betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                weight_decay=self.Optim_settings['weight_decay'])
            Adam_embedding = torch.optim.Adam([self.model.encode_embedding, self.model.encode_bias], lr=self.Optim_settings['lr_sghmc'],
                                              betas=(self.Optim_settings['beta1'], self.Optim_settings['beta2']),
                                              weight_decay=self.Optim_settings['weight_decay'])
        else:

            Adam_encoder = self.Adam_encoder
            Adam_embedding = self.Adam_embedding

        kwargs['Adam_encoder']=Adam_encoder
        kwargs['Adam_embedding']=Adam_embedding

        # Train the sghmc
        W_sample,_,_,_=train_SGHMC(self.Infer_model,observed_train,eps=eps,max_sample_size=max_sample_size,tot_epoch=tot_epoch,thinning=thinning,hyper_param_update=hyper_param_update,
                    sample_int=sample_int,flag_hybrid=flag_hybrid,flag_results=False,W_dict_init=W_dict_init,flag_imputation=flag_imputation,**kwargs
                    )
        return W_sample
    def random_select_test(self,**kwargs):
        # Test time random selection
        test_input = kwargs['test_input']  # If this is the first time run, this should be all initialized at 0
        pool_data_tensor = kwargs['pool_data_tensor']
        total_user = pool_data_tensor.shape[0]
        # random select
        idx_array = []
        for user_i in range(total_user):
            non_zero_idx = (torch.abs(pool_data_tensor[user_i, :]) > 0.).nonzero()  # 1D idx tensor

            select_idx = non_zero_idx[torch.randperm(len(non_zero_idx))[0]]  # random select 1 index
            test_input[user_i, select_idx] = pool_data_tensor[user_i, select_idx]
            # remove the pool
            pool_data_tensor[user_i, select_idx] = 0.
            idx_array.append(select_idx)
        return test_input,idx_array,pool_data_tensor
    def generate_active_learn_input_z_target_test(self,test_input,pool_candidate,target_candidate,flag_only_i=False,flag_same_pool=False,**kwargs):
        # test_input is the observed test_input, pool_candidate is the NW x Nz x N x obs after applying pool_mask, target_candidate is N_w x Nz x N x obs after applying target_candidate
        # flag_same_pool is to indicate whether for each user i, the pool has the same number of candidates (for fully observed pool, it is true, but if pool has some missing data, then this may not be true)
        # This flag is for efficient computation
        # flag_only_i indicate whether only generate for row i/slice i
        # This is used as an component for EDDI selection at test time

        if flag_only_i==True:
            active_input_o_target=kwargs['active_input_o_target']
        if flag_same_pool==False:
            slice=kwargs['slice']

        if flag_only_i==False:
            # decouple test_input
            test_input=test_input.clone().detach() # N x obs
            # Store N and obs
            N,obs=target_candidate.shape[2],target_candidate.shape[3]
            # Generate X_phi, X_o
            target_candidate_reshape=target_candidate.view(-1,target_candidate.shape[2],target_candidate.shape[3]) # tot x N x obs
            target_candidate_reshape=target_candidate_reshape.view(target_candidate_reshape.shape[0],-1) # tot x (N x obs)

            # Non-zero idx
            non_zero_idx=(torch.abs(target_candidate_reshape)>0.).nonzero() # 2D array
            non_zero_idx=non_zero_idx[:,1]
            non_zero_idx=torch.unique(non_zero_idx) # 1D array with size d_target

            total_sample_number = target_candidate_reshape.shape[0]

            non_zero_idx_expand = torch.unsqueeze(non_zero_idx, dim=0).repeat(total_sample_number,
                                                                                               1)  # tot x d_pool
            test_input_reshape=test_input.view(1,-1).repeat(total_sample_number,1) # tot x (Nxobs)

            # select index
            target_candidate_reshape_selected = torch.index_select(target_candidate_reshape, dim=1,
                                                                   index=non_zero_idx)  #TODO: Check (total) x d_target (Checked)


            active_input_o_target=test_input_reshape.scatter_(dim=1,index=non_zero_idx_expand,src=target_candidate_reshape_selected) # tot x (Nxobs)
            active_input_o_target=active_input_o_target.view(total_sample_number,N,obs) # TODO: Check tot x N x obs and Check if it does the desired behavior (Checked)

        if flag_same_pool==True:
            # Generate X_phi,X_o,X_i
            total_sample_number_pool=pool_candidate.shape[0]*pool_candidate.shape[1]
            d_active=torch.sum(torch.abs(pool_candidate[0,0,0,:])>0.).int()
            pool_candidate_reshape=pool_candidate.view(-1,N,obs) # tot x N x obs
            #print('d_active:%s'%(d_active))
            active_input_o_target_rep=torch.unsqueeze(active_input_o_target,dim=2).repeat(1,1,d_active,1) #TODO: Check if tot x N x d x obs

            non_zero_idx_pool=(torch.abs(pool_candidate_reshape)>0.).nonzero() # 3 eleement matrix
            non_zero_idx_pool=non_zero_idx_pool[:,1:] # 2D with N,pool_dim
            non_zero_idx_pool_reshape=non_zero_idx_pool[0:N*d_active,1:2].view(N,d_active) # TODO: Check if N x d (Checked)
            non_zero_idx_pool_reshape_final=torch.unsqueeze(torch.unsqueeze(non_zero_idx_pool_reshape,dim=0),dim=3).repeat(total_sample_number_pool,1,1,1) # TODO: Check if tot x N x d x 1

            pool_candidate_reshape_reshape=pool_candidate_reshape.view(total_sample_number_pool,-1) # tot x (N x obs)
            non_zero_idx_tmp=(torch.abs(pool_candidate_reshape_reshape)>0.).nonzero()
            non_zero_idx_tmp=torch.unique(non_zero_idx_tmp[:,1]) # 1 D array

            pool_candidate_reshape_reshape=torch.index_select(pool_candidate_reshape_reshape,dim=1,index=non_zero_idx_tmp) #TODO: Check tot x (N x d)
            pool_candidate_reshape_reshape=pool_candidate_reshape_reshape.view(-1,N,d_active)
            pool_candidate_reshape_reshape=torch.unsqueeze(pool_candidate_reshape_reshape,dim=3) # TODO: Check if tot x N x d x 1(Checked)
            active_input_o_target_i=active_input_o_target_rep.scatter_(dim=3,index=non_zero_idx_pool_reshape_final,src=pool_candidate_reshape_reshape) # TODO Check if does desired and shape tot x N x d x obs
            return active_input_o_target,active_input_o_target_i,non_zero_idx_pool_reshape
        else:
            active_input_o_target_slice=active_input_o_target[:,slice,:]
            pool_candidate_slice = pool_candidate.view(-1, pool_candidate.shape[2],
                                                       pool_candidate.shape[3])  # (N_zxN_W) x N x obs_dim
            pool_candidate_slice = torch.unsqueeze(pool_candidate_slice[:, slice, :],
                                                   dim=2)  # (total_sample_num) x obs_dim x 1
            non_zero_idx = (torch.abs(pool_candidate_slice) > 0.).nonzero()  # 3D tensor
            non_zero_idx_dim = non_zero_idx[:, 1]
            non_zero_idx_dim = torch.unique(non_zero_idx_dim)
            total_pool_size = non_zero_idx_dim.shape[0]
            total_sample_number = pool_candidate_slice.shape[0]
            # Non zero pool
            pool_candidate_slice = torch.index_select(pool_candidate_slice, dim=1,
                                                      index=non_zero_idx_dim)  # (total) x d_pool x 1
            # index array
            non_zero_idx_array = torch.unsqueeze(
                torch.unsqueeze(non_zero_idx_dim, dim=0).repeat(total_sample_number, 1),
                dim=2)  # N x d_pool x 1

            # replicate the active_input_target
            active_input_pool_target = torch.unsqueeze(active_input_o_target_slice, dim=1).repeat(1, total_pool_size,
                                                                                          1)  # tot x d x obs_dim

            active_input_pool_target = active_input_pool_target.scatter_(dim=2, index=non_zero_idx_array,
                                                                         src=pool_candidate_slice)  # total x d x obs_dim
            return active_input_o_target,active_input_pool_target,non_zero_idx_dim


    def generate_active_learn_input_z_i_test(self,test_input,pool_candidate,flag_same_pool=False,**kwargs):
        # Also used as a component for EDDI computation in test time
        if flag_same_pool==False:
            slice=kwargs['slice']
        # Decouple test_input
        test_input=test_input.clone().detach()
        if flag_same_pool==True:
            N,obs=test_input.shape[0],test_input.shape[1]
            d_active=torch.sum(torch.abs(pool_candidate[0,0,0,:])>0.).int()
            pool_candidate_reshape=pool_candidate.view(-1,N,obs) # tot x N x obs
            sample_tot=pool_candidate_reshape.shape[0]
            test_input_reshape=torch.unsqueeze(test_input,dim=1).repeat(1,d_active,1) # N x d x obs
            test_input_reshape=torch.unsqueeze(test_input_reshape,dim=0).repeat(sample_tot,1,1,1) # tot x N x d x obs

            non_zero_idx=(torch.abs(pool_candidate_reshape)>0.).nonzero()
            non_zero_idx=non_zero_idx[:,1:] # 2 D array
            non_zero_idx_reshape=non_zero_idx[0:N*d_active,1:2].view(N,d_active) # N x d_active
            non_zero_idx_reshape_final=torch.unsqueeze(torch.unsqueeze(non_zero_idx_reshape,dim=2),dim=0).repeat(sample_tot,1,1,1) # tot x N x d x 1

            pool_candidate_flat=pool_candidate_reshape.view(sample_tot,-1)
            non_zero_idx_tmp=(torch.abs(pool_candidate_flat)>0.).nonzero()
            non_zero_idx_tmp=torch.unique(non_zero_idx_tmp[:,1]) # 1 d array

            pool_candidate_select=torch.index_select(pool_candidate_flat,dim=1,index=non_zero_idx_tmp)
            pool_candidate_select=torch.unsqueeze(pool_candidate_select.view(sample_tot,N,d_active),dim=3) # tot x N x d x 1

            active_input_o_i=test_input_reshape.scatter_(dim=3,index=non_zero_idx_reshape_final,src=pool_candidate_select)

            return active_input_o_i,non_zero_idx_reshape



        else:
            # Generate the input for computing the BALD, slice indicate the user row number for test_input
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
            non_zero_idx_array = torch.unsqueeze(
                torch.unsqueeze(non_zero_idx_dim, dim=0).repeat(total_sample_number, 1), dim=2)  # N x d_pool x 1
            # replicate the test_input
            test_input_slice = test_input[slice, :]  # obs_dim
            test_input_slice = torch.unsqueeze(torch.unsqueeze(test_input_slice, dim=0), dim=0).repeat(
                total_sample_number, total_pool_size, 1)

            active_input = test_input_slice.scatter_(dim=2, index=non_zero_idx_array,
                                                     src=pool_candidate_slice)  # total x d x obs_dim

            return active_input, non_zero_idx_dim

    def active_learn_target_test(self,**kwargs):
        # EDDI computation and selection
        flag_same_pool=kwargs['flag_same_pool']

        test_input = kwargs['test_input']
        pool_data_tensor = kwargs['pool_data_tensor']
        target_data_tensor = kwargs['target_data_tensor']
        size_z=kwargs['size_z']
        W_sample=kwargs['W_sample']
        split=kwargs['split']
        test_input_orig=test_input.clone().detach()
        if split>1 and split<test_input.shape[0]:
            batch_size = int(math.ceil((test_input.shape[0] / split)))
            pre_idx = 0
            counter_idx = 0
            for idx in range(split + 1):
                idx = min((idx + 1) * batch_size, test_input.shape[0])
                if pre_idx == idx:
                    break

                data_input = test_input[pre_idx:idx, :]
                data_pool=pool_data_tensor[pre_idx:idx, :]
                data_target=target_data_tensor[pre_idx:idx, :]

                data_input_mask = get_mask(data_input)

                # Sample X_phi and X_id
                z, _ = self.model.sample_latent_variable(data_input, data_input_mask, size=size_z)  # size_z x N x latent
                if self.Infer_model.flag_LV:
                    decode,_=self.Infer_model.sample_X(z,W_sample)
                else:
                    decode = self.Infer_model.sample_X(z,W_sample) # N_w x Nz x N x obs

                # Remove memory of decode and z
                decode = decode.clone().detach()
                z = z.clone().detach()

                # Reduce size for memory reduction
                # perm1,perm2=torch.randperm(decode.shape[0])[0:20],torch.randperm(decode.shape[1])[0:5]
                #
                # decode=reduce_size(decode,perm1,perm2) # red_w x red_z x N x obs
                # z=reduce_size(z,perm1,perm2)# red_z x N x obs







                target_mask = get_mask(data_target)  # N x obs_dim

                target_mask_expand = torch.unsqueeze(torch.unsqueeze(target_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim

                target_candidate = target_mask_expand * decode  # N_w x N_z x N x obs_dim

                pool_mask = get_mask(data_pool)
                pool_mask_expand = torch.unsqueeze(torch.unsqueeze(pool_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim
                pool_candidate = pool_mask_expand * decode  # N_w x N_z x N x obs_dim
                total_user = pool_candidate.shape[2]


                if flag_same_pool:
                    # Efficient computation, compute values in parallel for all users
                    # sample (X_o,X_phi),(X_o,X_target),(X_o,X_target,X_i)
                    active_input_o_target,active_input_o_i_target,non_zero_idx=self.generate_active_learn_input_z_target_test(data_input,pool_candidate,target_candidate,flag_only_i=False,flag_same_pool=True)
                    active_input_o_i,non_zero_idx=self.generate_active_learn_input_z_i_test(data_input,pool_candidate,flag_same_pool=True)

                    mask_active_target = get_mask(active_input_o_target)  # tot x N x obs
                    mask_active_pool_target = get_mask(active_input_o_i_target)  # tot x N x d x obs_dim
                    mask_active = get_mask(active_input_o_i) # tot x N x d x obs

                    encoding_target = self.model._encoding(active_input_o_target,
                                                           mask_active_target)  # total x N x 2*latent
                    encoding_pool_target = self.model._encoding(active_input_o_i_target,
                                                                mask_active_pool_target)  # tot x N x d x 2*latent
                    encoding_o_i = self.model._encoding(active_input_o_i, mask_active)  # total x N x d x 2*latent

                    encoding_o= self.model._encoding(data_input, data_input_mask) # N x 2*latent

                    # clear memory
                    encoding_target = encoding_target.clone().detach()
                    encoding_pool_target = encoding_pool_target.clone().detach()
                    encoding_o_i = encoding_o_i.clone().detach()
                    encoding_o=encoding_o.clone().detach()

                    encoding_target = torch.unsqueeze(encoding_target, dim=2).repeat(1, 1,encoding_pool_target.shape[2],
                                                                                     1)  # tot x N x d x 2*latent
                    KL_2 = self._gaussian_KL_all(encoding_pool_target, encoding_target)  # tot x N x d

                    encoding_o=torch.unsqueeze(encoding_o,dim=0).repeat(encoding_o_i.shape[0],1,1)
                    encoding_o=torch.unsqueeze(encoding_o,dim=2).repeat(1,1,encoding_o_i.shape[2],1) # tot  x N x d x 2*latent
                    KL_1=self._gaussian_KL_all(encoding_o_i, encoding_o)  # tot x N x d

                    loss=KL_1-KL_2
                    mean_loss = torch.mean(loss, dim=0)  # N x d

                    # Take the maximum value

                    _,idx_max=torch.max(mean_loss,dim=1)

                    # Sample according to mean_loss
                    mean_scale = torch.mean(torch.abs(mean_loss),dim=1,keepdim=True) # N x 1
                    temp = (0.3 + 0) * mean_scale
                    mean_loss_prob = torch.clamp(
                        F.softmax(mean_loss / temp), max=1.,min=1e-3) # N x d

                    #idx_prob=torch.multinomial(mean_loss_prob, 1).view(-1) # N x 1

                    # original index
                    idx_selected=non_zero_idx[np.arange(non_zero_idx.shape[0]),idx_max] # N
                    #idx_selected=non_zero_idx[np.arange(non_zero_idx.shape[0]),idx_prob] # N

                    # update data_input
                    test_input[torch.arange(pre_idx,idx), idx_selected]=pool_data_tensor[torch.arange(pre_idx,idx), idx_selected]
                    # update the pool by removing the selected ones
                    pool_data_tensor[torch.arange(pre_idx,idx), idx_selected] = 0.



                else:
                    # less efficient computation, compute EDDI for single row at a time
                    active_input_o_target, _,_= self.generate_active_learn_input_z_target_test(
                        data_input, pool_candidate, target_candidate, flag_only_i=False, flag_same_pool=False,slice=0) # what ever slice is fine

                    for user_i in range(total_user):
                        _, active_input_o_i_target, non_zero_idx = self.generate_active_learn_input_z_target_test(
                            data_input, pool_candidate, target_candidate, flag_only_i=True, flag_same_pool=False,active_input_o_target=active_input_o_target,slice=user_i)
                        active_input_o_i, non_zero_idx = self.generate_active_learn_input_z_i_test(data_input,
                                                                                                   pool_candidate,
                                                                                                   flag_same_pool=False,slice=user_i)
                        active_input_o_target_slice=active_input_o_target[:,user_i,:]


                        mask_active_target = get_mask(active_input_o_target_slice)  # tot  x obs
                        mask_active_pool_target = get_mask(active_input_o_i_target)  # tot x d x obs_dim
                        mask_active = get_mask(active_input_o_i)  # tot x d x obs

                        encoding_target = self.model._encoding(active_input_o_target_slice,
                                                               mask_active_target)  # total x 2*latent
                        encoding_pool_target = self.model._encoding(active_input_o_i_target,
                                                                    mask_active_pool_target)  # tot  x d x 2*latent
                        encoding_o_i = self.model._encoding(active_input_o_i, mask_active)  # total  x d x 2*latent

                        encoding_o = self.model._encoding(torch.unsqueeze(data_input[user_i,:],dim=0), torch.unsqueeze(data_input_mask[user_i,:],dim=0))  # 1 x 2*latent

                        # clear memory
                        encoding_target = encoding_target.clone().detach()
                        encoding_pool_target = encoding_pool_target.clone().detach()
                        encoding_o_i = encoding_o_i.clone().detach()
                        encoding_o = encoding_o.clone().detach()

                        encoding_target = torch.unsqueeze(encoding_target, dim=1).repeat(1,
                                                                                         encoding_pool_target.shape[1],
                                                                                         1)  # tot x d x 2*latent
                        KL_2 = self._gaussian_KL(encoding_pool_target, encoding_target)  # tot x d

                        encoding_o = torch.unsqueeze(encoding_o, dim=0).repeat(encoding_o_i.shape[0], 1, 1) # tot x 1 x latent
                        encoding_o = encoding_o.repeat(1, encoding_o_i.shape[1],
                                                                               1)  # tot x d x 2*latent
                        KL_1 = self._gaussian_KL(encoding_o_i, encoding_o)  # tot  x d

                        loss = KL_1 - KL_2
                        mean_loss = torch.mean(loss, dim=0)  # d

                        _, idx_max = torch.max(mean_loss, dim=0)

                        # original index
                        idx_selected = non_zero_idx[idx_max]  # N
                        # update data_input
                        test_input[counter_idx, idx_selected] = pool_data_tensor[counter_idx, idx_selected]
                        # update the pool by removing the selected ones
                        pool_data_tensor[counter_idx, idx_selected] = 0.
                        counter_idx += 1
                pre_idx=idx


        else:
            raise NotImplementedError

        # Selection stat (Debug Purpose )
        choice_now=get_choice(test_input)
        choice_orig=get_choice(test_input_orig)
        choice_stat=choice_now-choice_orig
        return test_input,choice_stat,pool_data_tensor

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

    def _gaussian_KL_all(self,encoding1,encoding2):
        # assert if the shape matches, normally is tot x N x d x 2*latent
        assert encoding1.shape==encoding2.shape, ' Inconsistent encoding shapes'
        mean1,mean2=encoding1[:,:,:,0:self.model.latent_dim],encoding2[:,:,:,0:self.model.latent_dim] # tot x N x d x latent
        if self.model.flag_log_q==True:
            sigma1, sigma2 = torch.sqrt(torch.exp(encoding1[:,:, :, self.model.latent_dim:])), torch.sqrt(torch.exp(
                encoding2[:,:, :, self.model.latent_dim:]))  # tot x N x d x latent
        else:
            sigma1,sigma2=torch.sqrt(encoding1[:,:,:,self.model.latent_dim:]**2),torch.sqrt(encoding2[:,:,:,self.model.latent_dim:]**2) # tot x N x d x latent
        KL=torch.log(sigma2/sigma1)+(sigma1**2+(mean1-mean2)**2)/(2*sigma2**2)-0.5 # tot x N x d x latent
        KL=torch.sum(KL,dim=3) # tot x N x d
        return KL


class base_Active_Learning_SGHMC_Decoder(base_Active_Learning_SGHMC):
    def __init__(self,*args,**kwargs):
        super(base_Active_Learning_SGHMC_Decoder, self).__init__(*args, **kwargs)
    def get_target_variable(self,observed_train,observed_train_before,target_dim,train_data=None):
        observed_train=observed_train.clone().detach()
        diff_observed=torch.abs(observed_train-observed_train_before) # N x out_dim
        sum_diff=torch.sum(diff_observed,dim=1) # N
        # Apply the target variable
        if train_data is None:
            observed_train[sum_diff>0.,target_dim]=self.train_data_tensor[sum_diff>0.,target_dim] #Assume train_data_tensor has the same arrangement as observed_train
        else:
            observed_train[sum_diff > 0., target_dim] = train_data[sum_diff > 0., target_dim]
        return observed_train.clone().detach()
    def _transform_idx(self,idx,N,obs,num_unobserved,step):
        num_observed=idx.shape[1]-num_unobserved
        counter = 0
        if num_observed==0:
            idx=[]
            return [],True
        else:
            idx=idx[0,:num_observed] #remove the unobserved
            current_size=idx.shape[0]
            if current_size<=step:
                idx_selected=idx
            else:
                idx_selected=idx[0:step]
            row = (idx_selected / obs).view(-1, 1)
            column = (idx_selected % obs).view(-1, 1)
            return torch.cat((row, column), dim=1), False
    def _transform_idx_pure(self,idx,obs):
        # similar to _transform_idx but used as different components in different functions
        row = (idx / obs).view(-1, 1)
        column = (idx % obs).view(-1, 1)
        return torch.cat((row, column), dim=1)

    def _apply_selected_idx(self,observed_train,pool_data,idx):
        for i in range(idx.shape[0]):
            row=idx[i,0]
            column=idx[i,1]
            #Assign to observed train
            observed_train[row,column]=pool_data[row,column]
            # Remove pool_data
            pool_data[row,column]=0.
        return observed_train,pool_data
    def _H_Xid_Xo(self,decode,comp_mean,sigma_out,flag_reduce_size=False,**kwargs):
        # Compute H[p(x_id|x_o)]
        if flag_reduce_size==True:
            perm1, perm2 = torch.randperm(decode.shape[0])[0:10], torch.randperm(decode.shape[1])[0:5]

            decode = reduce_size(decode, perm1, perm2)  # red_w x red_z x N x obs
            comp_mean=reduce_size(comp_mean,perm1,perm2)
        N=decode.shape[2]
        obs_dim=decode.shape[3]
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # decode_LV=kwargs['decode_LV']
            # decode_LV=decode_LV.clone().detach()
            # decode_LV=decode_LV.view(-1,N,obs_dim)
        decode=decode.view(-1,N,obs_dim) # (N_W x N_z) x N x obs

        comp_mean=comp_mean.view(-1,N,obs_dim) #(N_w x N_z) x N x obs

        decode_exapnd=torch.unsqueeze(decode,dim=1) #  (NwNz) x 1 x N x obs
        comp_mean_expand=torch.unsqueeze(comp_mean,dim=0).repeat(comp_mean.shape[0],1,1,1)# (NwNz) x (NwNz) x N x obs


        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # comp_LV_expand=torch.unsqueeze(decode_LV,dim=0).repeat(decode_LV.shape[0],1,1,1)# (NwNz) x (NwNz) x N x obs
            # # Compute log likeihood
            # log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * torch.log(comp_LV_expand ** 2) - 1. / (2 * comp_LV_expand ** 2) * (
            #             decode_exapnd - comp_mean_expand) ** 2  # (NwNz) x (NwNz) x N x obs
        else:
            # Compute log likeihood
            log_likelihood=-0.5*math.log(2*np.pi)-0.5*math.log(sigma_out**2)-1./(2*sigma_out**2)*(decode_exapnd-comp_mean_expand)**2 # (NwNz) x (NwNz) x N x obs
        # Compute logsumexp
        marginal_log_likelihood=torch.logsumexp(log_likelihood,dim=1)-math.log(float(decode_exapnd.shape[0])) # (Nw_Nz) x N x obs
        entropy= -torch.mean(marginal_log_likelihood,dim=0) # N x obs
        # Clear memory
        entropy=torch.tensor(entropy.data)
        return entropy

    # def _H_X_id_W_X_phi_O(self,decode,observed_data,target_dim,W_sample):
    #     observed_data=torch.tensor(observed_data.data)
    #     # First sample X_Phi
    #     X_Phi = decode[:, :, :, target_dim]  # Nw x Nz x N
    #     X_Phi = X_Phi.view(-1, X_Phi.shape[2])  # N_phi x N
    #     # Remove memory
    #     X_Phi = X_Phi.clone().detach()
    #     # Now sample X_id|X_o,X_phi,W
    #     X_Phi_exp = torch.unsqueeze(X_Phi, dim=2)  # N_p x N x 1
    #     observed_data_exp = torch.unsqueeze(observed_data, dim=0).repeat(X_Phi.shape[0], 1, 1)  # N_p x N x obs
    #     idx = torch.tensor([target_dim]).long()  # 1
    #     idx_exp = torch.unsqueeze(torch.unsqueeze(idx, dim=0), dim=0).repeat(X_Phi.shape[0], observed_data.shape[0],
    #                                                                          1)  # N_p x N x 1
    #     observed_data_exp.scatter_(2, idx_exp, X_Phi_exp)  # N_p x N x obs
    #     observed_data_exp = observed_data_exp.view(-1, observed_data_exp.shape[2])  # (N_pN) x obs
    #     # Sample z
    #     mask_observed_data_exp = get_mask(observed_data_exp)
    #     z, _ = self.model.sample_latent_variable(observed_data_exp, mask_observed_data_exp, size=10)  # size_z x N_pN x latent
    #
    #     # Transform back
    #     z = z.view(z.shape[0], idx_exp.shape[0], idx_exp.shape[1], z.shape[2])  # size_z x Np x N x latent
    #     z = z.clone().detach()
    #
    #
    #     if self.Infer_model.flag_LV:
    #         decode,decode_LV=self.Infer_model.sample_X(z,W_sample) # Nw x nz x np x n x obs
    #         decode_LV=decode_LV.clone().detach()
    #         # Convert to sigma
    #         decode_LV=torch.sqrt(torch.exp(decode_LV))
    #     else:
    #         decode = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x Np x N x obs
    #     # _, decode = self.model.decoding(z, self.sigma_out, flag_record=False,
    #     #                                 size_W=5)  # decode with shape N_w x N_z x Np x N x obs_dim
    #     decode = decode.clone().detach()
    #     # Compute log likleihood
    #     comp_mean = torch.tensor(decode.data) # Nw x Nz x Np x N x obs
    #     decode_exapnd = torch.unsqueeze(decode, dim=2)  # Nw x Nz x 1 x Np x N x obs
    #     comp_mean_expand = torch.unsqueeze(comp_mean, dim=1).repeat(1,comp_mean.shape[1], 1, 1,
    #                                                                 1, 1)  # Nw x Nz(copy) x Nz x Np x N x obs
    #     if self.Infer_model.flag_LV:
    #         comp_LV_exp=torch.unsqueeze(decode_LV, dim=1).repeat(1,comp_mean.shape[1], 1, 1,
    #                                                                 1, 1)  # Nw x Nz(copy) x Nz x Np x N x obs
    #         log_likelihood=-0.5 * math.log(2 * np.pi) - 0.5 * torch.log(comp_LV_exp ** 2) - 1. / (
    #                 2 * comp_LV_exp ** 2) * (
    #                              decode_exapnd - comp_mean_expand) ** 2  # Nw x Nz (sample) x Nz(comp) x Np x N x obs
    #     else:
    #         log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
    #                 2 * self.sigma_out ** 2) * (
    #                              decode_exapnd - comp_mean_expand) ** 2  # Nw x Nz (sample) x Nz(comp) x Np x N x obs
    #     marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=2) - math.log(
    #         float(decode_exapnd.shape[1]))  # Nw x Nz x Np x N x obs
    #     entropy = -torch.mean(marginal_log_likelihood, dim=1) # N w x Np x N x obs
    #     E_entropy=torch.mean(torch.mean(entropy,dim=0),dim=0)
    #     return torch.tensor(E_entropy.data)
    # def _H_X_id_X_phi_O(self,decode,observed_data,target_dim,W_sample):
    #     observed_data=torch.tensor(observed_data.data)
    #     # Reduce size
    #     perm1,perm2=torch.randperm(decode.shape[0])[0:20],torch.randperm(decode.shape[1])[0:5]
    #
    #     decode=reduce_size(decode,perm1,perm2) # red_w x red_z x N x obs
    #     # First sample X_Phi
    #     X_Phi=decode[:,:,:,target_dim] # Nw x Nz x N
    #     X_Phi=X_Phi.view(-1,X_Phi.shape[2]) # N_phi x N
    #     # Remove memory
    #     X_Phi=X_Phi.clone().detach()
    #     # Now sample X_id|X_o,X_phi
    #     # Form input
    #     X_Phi_exp=torch.unsqueeze(X_Phi,dim=2) # N_p x N x 1
    #     observed_data_exp=torch.unsqueeze(observed_data,dim=0).repeat(X_Phi.shape[0],1,1) # N_p x N x obs
    #     idx=torch.tensor([target_dim]).long() # 1
    #     idx_exp=torch.unsqueeze(torch.unsqueeze(idx,dim=0),dim=0).repeat(X_Phi.shape[0],observed_data.shape[0],1) # N_p x N x 1
    #     observed_data_exp.scatter_(2,idx_exp,X_Phi_exp) # N_p x N x obs
    #     observed_data_exp=observed_data_exp.view(-1,observed_data_exp.shape[2]) # (N_pN) x obs
    #     # Sample z
    #     mask_observed_data_exp = get_mask(observed_data_exp)
    #     z, _ = self.model.sample_latent_variable(observed_data_exp, mask_observed_data_exp,
    #                                              size=10)  # size_z x N_pN x latent
    #
    #     # Transform back
    #     z = z.view(z.shape[0], idx_exp.shape[0], idx_exp.shape[1], z.shape[2])  # size_z x Np x N x latent
    #     z = z.clone().detach()
    #
    #     if self.Infer_model.flag_LV:
    #         decode,decode_LV=self.Infer_model.sample_X(z,W_sample) # Nw x nz x np x n x obs
    #         decode_LV=decode_LV.clone().detach()
    #         # Convert to sigma
    #         decode_LV=torch.sqrt(torch.exp(decode_LV))
    #     else:
    #         decode = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x Np x N x obs
    #
    #     # Reduce sample size
    #     perm1, perm2 = torch.randperm(decode.shape[0])[0:10], torch.randperm(decode.shape[1])[0:5]
    #
    #     decode = reduce_size(decode, perm1, perm2)  # red_w x red_z x Np x N x obs
    #
    #     if self.Infer_model.flag_LV:
    #         decode_LV=reduce_size(decode_LV,perm1,perm2) # red_w x red_z x Np x N x obs
    #
    #
    #
    #     decode=decode.clone().detach()
    #
    #     # Compute likleihood
    #
    #     decode=decode.view(-1,decode.shape[2],decode.shape[3],decode.shape[4]) # (NwNz) x Np x N x obs
    #     comp_mean=torch.tensor(decode.data)
    #     decode_exapnd = torch.unsqueeze(decode, dim=1)  # (NwNz) x 1 x Np x N x obs
    #     comp_mean_expand = torch.unsqueeze(comp_mean, dim=0).repeat(comp_mean.shape[0], 1, 1,
    #                                                                 1,1)  # (NwNz) x (NwNz) x Np x N x obs
    #     if self.Infer_model.flag_LV:
    #         decode_LV=decode_LV.view(-1,decode_LV.shape[2],decode_LV.shape[3],decode_LV.shape[4])
    #         comp_LV_expand=torch.unsqueeze(decode_LV, dim=0).repeat(decode_LV.shape[0], 1, 1,
    #                                                                 1,1) # (NwNz) x (NwNz) x Np x N x obs
    #         log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * torch.log(comp_LV_expand ** 2) - 1. / (
    #                     2 * comp_LV_expand ** 2) * (
    #                                  decode_exapnd - comp_mean_expand) ** 2  # (NwNz) x (NwNz) x Np x N x obs
    #     else:
    #         log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (2 * self.sigma_out ** 2) * (
    #                 decode_exapnd - comp_mean_expand) ** 2  # (NwNz) x (NwNz) x Np x N x obs
    #     # Compute logsumexp
    #     marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=1) - math.log(
    #         float(decode_exapnd.shape[0]))  # (Nw_Nz) x Np x N x obs
    #     entropy = -torch.mean(marginal_log_likelihood, dim=0)  # Np x  N x obs
    #     # clear memory
    #     entropy=torch.tensor(entropy.data)
    #     return torch.mean(entropy,dim=0) # N x obs


    def _H_X_id_W_Xo(self,decode,comp_mean,sigma_out,flag_reduce_size=False,**kwargs): # Compute H[p(X_id|X_o,W)]
        # decode has shape nw x nz x n x obs
        # Reshape
        if flag_reduce_size==True:

            perm1, perm2 = torch.randperm(decode.shape[0])[0:10], torch.randperm(decode.shape[1])[0:5]

            decode = reduce_size(decode, perm1, perm2)  # red_w x red_z x N x obs
            comp_mean = reduce_size(comp_mean, perm1, perm2)

        N = decode.shape[2]
        obs_dim = decode.shape[3]
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # comp_LV=kwargs['decode_LV'] # nw x nz x n x obs
            # comp_LV=comp_LV.clone().detach()
            # comp_LV_expand = torch.unsqueeze(comp_LV, dim=1).repeat(1, comp_mean.shape[1], 1,
            #                                                             1, 1)  # N_W x N_z x N_z x N x obs

        decode_exapnd = torch.unsqueeze(decode, dim=2)  # N_w x N_z x 1 x N x obs
        comp_mean_expand = torch.unsqueeze(comp_mean, dim=1).repeat(1, comp_mean.shape[1], 1,
                                                                    1,1)  # N_W x N_z x N_z x N x obs
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # # Compute log likelihood
            # log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * torch.log(comp_LV_expand ** 2) - 1. / (2 * comp_LV_expand ** 2) * (
            #             decode_exapnd - comp_mean_expand) ** 2  # N_w x N_z x N_z x N x obs

        else:
            # Compute log likelihood
            log_likelihood=-0.5*math.log(2*np.pi)-0.5*math.log(sigma_out**2)-1./(2*sigma_out**2)*(decode_exapnd-comp_mean_expand)**2 # N_w x N_z x N_z x N x obs

        # Compute logsumexp
        marginal_log_likelihood=torch.logsumexp(log_likelihood,dim=2)-math.log(float(decode_exapnd.shape[1])) # N_w x N_z x N x obs
        entropy=-torch.mean(marginal_log_likelihood,dim=1) # N_w x N x obs
        entropy=torch.tensor(entropy.data)
        return entropy

    def _H_X_Phi_O_id(self, decode, observed_data_row, W_sample,row_idx,target_dim=-1):
        # Remove memory
        observed_data_row=observed_data_row.clone().detach() # N x obs
        decode=decode.clone().detach() # N_w x N_z x N x obs
        decode=decode[:,:,row_idx,:] # N_w x N_z x obs
        perm1, perm2 = torch.randperm(decode.shape[0])[0:5], torch.randperm(decode.shape[1])[0:5]
        decode = reduce_size(decode, perm1, perm2,flag_one_row=True)  # red_w x red_z  x obs
        decode=decode.view(-1,decode.shape[-1]) # tot x obs
        tot=decode.shape[0]
        # Get vacance location
        idx = (torch.abs(observed_data_row[0:-1]) == 0).nonzero().view(-1)  # N_id (get rid of target variable)
        if idx.shape[0] == 0:
            return 0 * torch.ones(observed_data_row.shape)
        size_idx=idx.shape[0] # N_pool
        observed_data_row_dup=torch.unsqueeze(torch.unsqueeze(observed_data_row,dim=0),dim=0).repeat(tot,size_idx,1) # tot x N_pool x obs
        idx_dup=torch.unsqueeze(torch.unsqueeze(idx,dim=0).repeat(tot,1),dim=2) # tot x N_pool x 1
        mask_candidate=1-get_mask(observed_data_row)
        mask_candidate[-1]=0
        decode=decode*torch.unsqueeze(mask_candidate,dim=0) # tot x obs with zero's
        decode=torch.t(remove_zero_row_2D(torch.t(decode))) # tot x N_pool
        decode_dup=torch.unsqueeze(decode,dim=2) # tot x N_pool x 1
        observed_data_row_dup.scatter_(2,idx_dup,decode_dup) # tot x N_pool x obs

        # Sample X_phi
        observed_data_row_dup = observed_data_row_dup.view(-1, observed_data_row_dup.shape[2])  # (totxN_pool) x obs
        # Sample z
        mask_observed_data_exp = get_mask(observed_data_row_dup)
        z, _ = self.model.sample_latent_variable(observed_data_row_dup, mask_observed_data_exp,
                                                 size=10)  # size_z x (totxN_pool) x latent

        # Transform back
        z = z.view(z.shape[0], tot, size_idx, z.shape[2])  # size_z x tot x N_pool x latent
        z = z.clone().detach()

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # decode_,decode_LV=self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x tot x N_pool x obs
            # decode_LV=decode_LV.clone().detach()
            # decode_LV=torch.sqrt(torch.exp(decode_LV))
        else:

            decode_ = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x tot x N_pool x obs
        decode_=decode_.clone().detach()
        # Reduce sample size to save memory
        perm1, perm2 = torch.randperm(decode_.shape[0])[0:10], torch.randperm(decode_.shape[1])[0:5]

        decode_ = reduce_size(decode_, perm1, perm2)  # red_w x red_z x tot x N_pool x obs
        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # decode_LV=reduce_size(decode_LV,perm1,perm2)
            # decode_LV=decode_LV[:,:,:,:,target_dim]# red_w x red_z x tot x N_pool
            # decode_LV=decode_LV.view(-1,decode_LV.shape[2],decode_LV.shape[3])# (w x z) x tot x N_pool
            # comp_LV_exp=torch.unsqueeze(decode_LV,dim=0).repeat(decode_LV.shape[0],1,1,1) # (wxz) x (wxz) x tot x N_pool
        decode_phi=decode_[:,:,:,:,target_dim] # red_w x red_z x tot x N_pool


        decode_phi=decode_phi.view(-1,decode_phi.shape[2],decode_phi.shape[3]) # (w x z) x tot x N_pool

        comp_mean=torch.unsqueeze(decode_phi,dim=0).repeat(decode_phi.shape[0],1,1,1) # (wxz) x (wxz) x tot x N_pool
        decode_phi_exp=torch.unsqueeze(decode_phi,dim=1) # (wxz) x 1 x tot x N_pool

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
            # log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * torch.log(comp_LV_exp ** 2) - 1. / (
            #         2 * comp_LV_exp ** 2) * (
            #                          decode_phi_exp - comp_mean) ** 2  # (wxz) x (wxz) x tot x N_pool
        else:

            log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
                2 * self.sigma_out ** 2) * (
                                 decode_phi_exp - comp_mean) ** 2  # (wxz) x (wxz) x tot x N_pool
        marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=1) - math.log(
            float(decode_phi_exp.shape[0]))  # (w x z) x tot x N_pool
        entropy = -torch.mean(marginal_log_likelihood, dim=0)  # tot x N_pool
        E_entropy_value = torch.mean(entropy,dim=0) # N_pool
        E_entropy = 0 * torch.ones(observed_data_row.shape)  # obs
        E_entropy[idx] = E_entropy_value  # obs
        return E_entropy
    def _H_X_Phi_W_id_O(self,decode,observed_data_row,W_sample,row_idx,target_dim=-1):
        # Remove memory
        observed_data_row = observed_data_row.clone().detach()  # N x obs
        decode = decode.clone().detach()  # N_w x N_z x N x obs
        decode = decode[:, :, row_idx, :]  # N_w x N_z x obs
        perm1, perm2 = torch.randperm(decode.shape[0])[0:5], torch.randperm(decode.shape[1])[0:5]
        decode = reduce_size(decode, perm1, perm2, flag_one_row=True)  # red_w x red_z  x obs
        decode = decode.view(-1, decode.shape[-1])  # tot x obs
        tot = decode.shape[0]
        # Get vacance location
        idx = (torch.abs(observed_data_row[0:-1]) == 0).nonzero().view(-1)  #  N_id (get rid of target variable)
        if idx.shape[0] == 0:
            # just set a large value to aviod being selected
            return 10e5 * torch.ones(observed_data_row.shape)

        size_idx = idx.shape[0]  # N_pool
        observed_data_row_dup = torch.unsqueeze(torch.unsqueeze(observed_data_row, dim=0), dim=0).repeat(tot, size_idx,
                                                                                                         1)  # tot x N_pool x obs
        idx_dup = torch.unsqueeze(torch.unsqueeze(idx, dim=0).repeat(tot,1),dim=2)  # tot x N_pool x 1
        mask_candidate = 1 - get_mask(observed_data_row)
        mask_candidate[-1] = 0
        decode = decode * torch.unsqueeze(mask_candidate, dim=0)  # tot x obs with zero's
        decode = torch.t(remove_zero_row_2D(torch.t(decode)))  # tot x N_pool
        decode_dup = torch.unsqueeze(decode, dim=2)  # tot x N_pool x 1
        observed_data_row_dup.scatter_(2, idx_dup, decode_dup)  # tot x N_pool x obs
        # Sample X_phi
        observed_data_row_dup = observed_data_row_dup.view(-1, observed_data_row_dup.shape[2])  # (totxN_pool) x obs
        # Sample z
        mask_observed_data_exp = get_mask(observed_data_row_dup)
        z, _ = self.model.sample_latent_variable(observed_data_row_dup, mask_observed_data_exp,
                                                 size=10)  # size_z x (totxN_pool) x latent

        # Transform back
        z = z.view(z.shape[0], tot, size_idx, z.shape[2])  # size_z x tot x N_pool x latent
        z = z.clone().detach() # P(z|X_id,X_O)

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            decode_ = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x tot x N_pool x obs
        decode_ = decode_.clone().detach()
        # Reduce sample size
        perm1, perm2 = torch.randperm(decode_.shape[0])[0:20], torch.randperm(decode_.shape[1])[0:5]

        decode_ = reduce_size(decode_, perm1, perm2)  # red_w x red_z x tot x N_pool x obs

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')

        decode_phi = decode_[:, :, :, :, target_dim]  # red_w x red_z x tot x N_pool

        comp_mean = torch.unsqueeze(decode_phi, dim=1).repeat(1,decode_phi.shape[1], 1, 1,
                                                              1)  # W x z x z x tot x N_pool
        decode_phi_exp = torch.unsqueeze(decode_phi, dim=2)  # W x z x 1 x tot x N_pool

        if self.Infer_model.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
                2 * self.sigma_out ** 2) * (
                                 decode_phi_exp - comp_mean) ** 2  # w x z x z x tot x N_pool

        marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=2) - math.log(
            float(decode_phi_exp.shape[1]))  # w x z x tot x N_pool

        entropy = -torch.mean(marginal_log_likelihood, dim=1)  # w x tot x N_pool
        E_entropy_value = torch.mean(torch.mean(entropy, dim=0),dim=0)  # N_pool
        E_entropy = 10e5 * torch.ones(observed_data_row.shape)  # obs
        E_entropy[idx] = E_entropy_value  # obs
        return E_entropy

    def base_active_learning_decoder(self,coef=0.999,balance_prop=0.25,temp_scale=0.25,flag_initial=False,split=50,strategy='Alpha',Select_Split=True,**kwargs):
        # balance_prop is the % that we selected from the already observed users
        observed_train=kwargs['observed_train']
        pool_data=kwargs['pool_data']
        step=kwargs['step'] #This is to define how many data points to select for each active learning
        sigma_out=kwargs['sigma_out']
        W_sample=kwargs['W_sample']
        sample_z_num = 10

        selection_scheme=kwargs['selection_scheme']

        if selection_scheme=='overall':
            print('%s selection scheme with Select_Split %s'%(selection_scheme,Select_Split))
        else:
            raise NotImplementedError

        if split>1 and split<pool_data.shape[0]:
            # split is to reduce the memory consumption
            batch_size = int(math.ceil(pool_data.shape[0] / split))
            pre_idx = 0
            counter=0
            for counter_idx in range(split + 1):
                idx = min((counter_idx + 1) * batch_size, pool_data.shape[0])
                if pre_idx == idx:
                    # finished computing BALD value for all pool data
                    break


                pool_data_split=pool_data[pre_idx:idx,:]
                observed_train_split=observed_train[pre_idx:idx,:]
                observed_train_mask=get_mask(observed_train_split)

                # Sample X_id
                z, _ = self.model.sample_latent_variable(observed_train_split, observed_train_mask,
                                                         size=sample_z_num)  # size_z x N_pN x latent

                if self.Infer_model.flag_LV:
                    raise RuntimeError('flag_LV must be False')
                else:
                    decode = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x N x obs

                # Remove memory
                z=z.clone().detach()
                decode=decode.clone().detach()

                if strategy=='Opposite_Alpha':
                    if self.Infer_model.flag_LV:
                        raise RuntimeError('flag_LV must be False')

                    else:
                        H_xid_xo = self._H_Xid_Xo(decode, decode.clone().detach(), sigma_out)  # N x obs

                        ############### Sum of I ########################
                        # Compute H[p(x_id|W,x_o)]
                        H_xid_w_xo = self._H_X_id_W_Xo(decode, torch.tensor(decode.data), sigma_out,
                                                   split=None)  # N_w x N x obs
                    # Compute E[H[p(x_id|W,x_o)]]
                    E_H_xid_w_xo = torch.mean(H_xid_w_xo, dim=0)  # N x obs
                    for row_id in range(observed_train_split.shape[0]):
                        E_H_Phi_id=self._H_X_Phi_O_id(decode,observed_train_split[row_id,:],W_sample,row_id) # obs
                        E_H_Phi_W_id=self._H_X_Phi_W_id_O(decode,observed_train_split[row_id,:],W_sample,row_id) # obs
                        if row_id==0:
                            E_loss_2=torch.unsqueeze(E_H_Phi_id-E_H_Phi_W_id,dim=0)
                        else:
                            E_loss_2_comp = torch.unsqueeze(E_H_Phi_id-E_H_Phi_W_id, dim=0)
                            E_loss_2 = torch.cat((E_loss_2, E_loss_2_comp), dim=0)

                    BALD_split = coef * (H_xid_xo - E_H_xid_w_xo) + (1. - coef) * E_loss_2  # N_split x obs
                elif strategy=='MC':

                    if self.Infer_model.flag_LV:
                        raise RuntimeError('flag_LV must be False')

                    else:
                        with torch.no_grad():
                            H_xid_xo = self._H_Xid_Xo(decode, decode.clone().detach(), sigma_out, split=None,flag_reduce_size=True)  # N x obs

                            ############### Sum of I ########################
                            # Compute H[p(x_id|W,x_o)]
                            H_xid_w_xo = self._H_X_id_W_Xo(decode, torch.tensor(decode.data), sigma_out,
                                                   split=None,flag_reduce_size=True)  # N_w x N x obs
                            E_H_xid_w_xo = torch.mean(H_xid_w_xo, dim=0)  # N x obs
                    BALD_split = (H_xid_xo - E_H_xid_w_xo)
                else:
                    raise NotImplementedError('strategy must be Opposite_Alpha or MC')

                ###################################################

                if counter==0:
                    BALD=BALD_split.clone().detach()
                else:
                    BALD=torch.cat((BALD,BALD_split),dim=0)
                counter+=1
                pre_idx=idx
        else:

            raise NotImplementedError


        # Clear memory
        BALD=BALD.clone().detach()
        pool_mask=get_mask(pool_data)
        if selection_scheme=='overall':
            if not flag_initial:
                # instead of selecting top K
                print('%s Exploration'%(self.model_name))

                # Sample according to BALD
                pool_mask_Byte = 1 - pool_mask.byte()



                ########## No new user exploration ########
                #BALD = assign_zero_row_2D_with_target(BALD,observed_train)  # In exploration, no new user can be selected, only select entries from the observed users
                ###########################################
                if Select_Split==True:
                    # Select_Split: Some of the features are selected from old users, others are selected from new users
                    # these numbers are controlled by balance_prop
                    BALD[pool_mask_Byte] = 0.
                    BALD[BALD < 0.] = 0.
                    step_new=int((1.-balance_prop)*step)

                    # Select from the new user
                    # Apply softmax

                    # Tempering a little
                    BALD_unobserved = assign_zero_row_2D_with_target_reverse(BALD, observed_train)
                    mean_scale=torch.mean(BALD_unobserved[BALD_unobserved>0.])
                    temp=(temp_scale+0.35)*mean_scale
                    BALD_unobserved[BALD_unobserved>0.]=torch.clamp(F.softmax(BALD_unobserved[BALD_unobserved>0.]/temp),max=1.,min=1e-10)

                    BALD_weight_unob=torch.squeeze(BALD_unobserved.view(1,-1))
                    idx_un, flag_full_un,select_num_un = BALD_Select_Explore(BALD_weight_unob, step_new)
                    if not flag_full_un:
                        idx_selected_unobserved = self._transform_idx_pure(idx_un, BALD.shape[1])
                        step_old=step-select_num_un
                    else:
                        # pool data has not remaining users, thus, all selected from the old users
                        print('BALD No more new user')
                        step_old=step-select_num_un

                    # Select from the observed
                    BALD_observed = assign_zero_row_2D_with_target(BALD, observed_train)
                    mean_scale=torch.mean(BALD_observed[BALD_observed>0.])
                    temp=temp_scale*mean_scale
                    # Apply softmax
                    BALD_observed[BALD_observed>0.]=torch.clamp(F.softmax(BALD_observed[BALD_observed>0.]/temp),max=1.,min=1e-10)

                    BALD_weight_ob=torch.squeeze(BALD_observed.view(1,-1))

                    idx_ob,flag_full_ob,select_num_ob=BALD_Select_Explore(BALD_weight_ob,step_old)

                    if not flag_full_ob:
                        # Still has remaining data in pool
                        idx_selected_observed=self._transform_idx_pure(idx_ob,BALD.shape[1])
                    else:
                        print('BALD No more old user')
                    # Concat together
                    if not flag_full_un:
                        flag_full = False
                        if not flag_full_ob:
                            idx_selected = torch.cat((idx_selected_unobserved, idx_selected_observed), dim=0)
                        else:
                            idx_selected = idx_selected_unobserved
                    elif not flag_full_ob:
                        flag_full = False
                        if not flag_full_un:
                            idx_selected = torch.cat((idx_selected_unobserved, idx_selected_observed), dim=0)
                        else:
                            idx_selected = idx_selected_observed
                    else:
                        print('No possible candidate')
                        flag_full = True
                else:
                    # select from pool, this does not distingush old/new users
                    BALD[pool_mask_Byte] = 0.
                    BALD[BALD < 0.] = 0.
                    BALD_ = BALD.clone().detach()
                    mean_scale = torch.mean(BALD_[BALD_ > 0.])
                    temp = (temp_scale + 0.) * mean_scale
                    BALD_[BALD_ > 0.] = torch.clamp(
                        F.softmax(BALD_[BALD_ > 0.] / temp), max=1.,min=1e-10)
                    BALD_weight_=torch.squeeze(BALD_.view(1,-1))

                    idx_, flag_full, select_num_ = BALD_Select_Explore(BALD_weight_, step)

                    if not flag_full:
                        idx_selected = self._transform_idx_pure(idx_, BALD.shape[1])
                    else:
                        print('BALD No more remaining pool data')


            else:
                # Select top K values initially instead of sampling
                pool_mask_Byte = 1 - pool_mask.byte()
                BALD[pool_mask_Byte] = 0.
                BALD[BALD < 0.] = 0.
                pool_mask_Byte = 1 - pool_mask.byte()
                BALD[pool_mask_Byte] = -(10e8)

                idx, num = BALD_Select(BALD)

                idx_selected, flag_full = self._transform_idx(idx, BALD.shape[0], BALD.shape[1], num,
                                                              step=step)  # (Nxobs) x 2


            # Apply to observed_train
            if not flag_full:
                num_selected = idx_selected.shape[0]
                observed_train,pool_data=self._apply_selected_idx(observed_train,pool_data,idx_selected)
            else:
                print('Full train data selected')
                num_selected=0
        else:
            raise NotImplementedError

        return observed_train,pool_data,flag_full,num_selected

    def base_BALD_select_row(self, split=30, **kwargs):
        print('Strong Baseline by traditional BALD')
        step = kwargs['step']  # this is the number of points to select
        pool_data = kwargs['pool_data']
        W_sample = kwargs['W_sample']
        # copy
        pool_data_cp = pool_data.clone().detach()
        observed_train = kwargs['observed_train']

        # now compute the BALD
        if split > 1 and split < pool_data.shape[0]:
            batch_size = int(math.ceil(pool_data.shape[0] / split))
            pre_idx = 0
            counter = 0
            for counter_idx in range(split + 1):
                idx = min((counter_idx + 1) * batch_size, pool_data.shape[0])
                if pre_idx == idx:
                    break

                pool_data_split = pool_data_cp[pre_idx:idx, :]
                observed_train_split = observed_train[pre_idx:idx, :]
                # Get pool data mask
                pool_mask = get_mask(pool_data_split)

                # compute p(y|x_i) where x_i from pool data
                z, _ = self.model.sample_latent_variable(pool_data_split, pool_mask,
                                                         size=10)  # size_z x N_split x latent
                # now sample y|z,w
                decode = self.Infer_model.sample_X(z, W_sample)  # N_w x Nz x N x obs
                # Decode label
                decode_label = torch.index_select(decode, -1,
                                                  torch.tensor([decode.shape[-1] - 1]))  # nw x nz x n x 1
                comp_mean = decode_label.clone().detach()
                decode_label_flat = torch.unsqueeze(
                    decode_label.view(-1, decode_label.shape[2], decode_label.shape[3]), dim=1)  # nwnz x 1 x n x 1
                comp_mean_flat = torch.unsqueeze(comp_mean.view(-1, decode_label.shape[2], decode_label.shape[3]),
                                                 dim=0)  # 1 x nwnz x n x 1

                # now compute marginal p(y_in|x_in)
                log_likelihood = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
                        2 * self.sigma_out ** 2) * (
                                         decode_label_flat - comp_mean_flat) ** 2  # nwnz x nwnz x n x 1
                marginal_log_likelihood = torch.logsumexp(log_likelihood, dim=1) - math.log(
                    float(log_likelihood.shape[0]))  # nwnz x n x 1
                # Compute entropy H[p(y_|x_i)]
                entropy = -torch.mean(marginal_log_likelihood, dim=0)  # n x 1

                # Now compute H[p(y_i,x_i,W)]

                decode_label_exp = torch.unsqueeze(decode_label, dim=2)  # nw x nz x 1 x n x 1
                comp_mean_exp = torch.unsqueeze(comp_mean, dim=1)  # nw x 1 x nz x n x 1
                # now compute marginal p(y_in|x_in,W)
                log_likelihood_W = -0.5 * math.log(2 * np.pi) - 0.5 * math.log(self.sigma_out ** 2) - 1. / (
                        2 * self.sigma_out ** 2) * (
                                           decode_label_exp - comp_mean_exp) ** 2  # nw x nz x nz x n x 1
                marginal_log_likelihood_W = torch.logsumexp(log_likelihood_W, dim=2) - math.log(
                    float(log_likelihood_W.shape[2]))  # nw x nz x n x 1
                # now compute H[p(y_i|x_i,W)]
                entropy_W = -torch.mean(marginal_log_likelihood_W, dim=1)  # nw x n x 1
                E_entropy = torch.mean(entropy_W, dim=0)  # n x 1
                BALD_split = entropy - E_entropy  # n x 1

                if counter == 0:
                    BALD = BALD_split.clone().detach()
                else:
                    BALD = torch.cat((BALD, BALD_split), dim=0)  # N_tot x 1

                counter += 1
                pre_idx = idx
        else:
            raise NotImplementedError

        # Now active select
        # excluding the observed train
        non_zero_row_idx = torch.sum(get_mask(pool_data_cp), dim=1).nonzero()  # N_non_zero
        zero_row_idx = (torch.sum(get_mask(pool_data_cp), dim=1) == 0).nonzero()  # N_zero
        accumu_data_points = 0
        if non_zero_row_idx.nelement() != 0:
            BALD = torch.squeeze(BALD)  # N_tot
            BALD[zero_row_idx] = -(1e5)

            _, idx_select = torch.sort(BALD, descending=True)

            for num_row in range(idx_select.shape[0]):
                row_idx = idx_select[num_row]
                num_feature_row = torch.sum(torch.abs(pool_data[row_idx, :]) > 0)
                observed_train[row_idx, :] = pool_data[row_idx, :]
                pool_data[row_idx, :] = 0.
                accumu_data_points += num_feature_row
                if accumu_data_points > step:
                    break
            return observed_train, pool_data, False
        else:
            print('No more Pool data for row selection')
            return observed_train, pool_data, True


    def base_random_select_training(self,balance_prop=0.25,Select_Split=True,flag_initial=False,**kwargs):
        # Random feature-wise selection
        observed_train = kwargs['observed_train']
        pool_data = kwargs['pool_data']
        pool_data_cp=torch.tensor(pool_data.data)
        step = kwargs['step']  # This is to define how many data points to select for each active learning

        obs=observed_train.shape[1]
        selection_scheme=kwargs['selection_scheme']
        if selection_scheme=='overall':
            print('%s selection scheme with Select_Split %s'%(selection_scheme,Select_Split))
        else:
            raise NotImplementedError
        if selection_scheme=='overall':
            if not flag_initial:
                if Select_Split==True:
                    print('%s Explore'%(self.model_name))

                    step_new=int((1-balance_prop)*step)
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
                    idx_ob,num_selected_ob,flag_full_ob=Random_select_idx(pool_data_cp_ob,obs,step_old)

                    if flag_full_ob:
                        print('No more Old user')
                    if not flag_full_un:
                        flag_full=False
                        if not flag_full_ob:
                            idx = torch.cat((idx_ob, idx_un), dim=0)
                        else:
                            idx=idx_un
                    elif not flag_full_ob:
                        flag_full = False
                        if not flag_full_un:
                            idx = torch.cat((idx_ob, idx_un), dim=0)
                        else:
                            idx = idx_ob
                    else:
                        flag_full=True

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

