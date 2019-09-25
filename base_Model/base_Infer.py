import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from base_Model.base_Network import *
from base_Model.base_BNN import *
from base_Model.BNN_Network_zoo import *
from Util.Util_func import *
from Dataloader.base_Dataloader import *
from torch.utils.data import DataLoader
from scipy.stats import bernoulli
from sklearn.utils import shuffle

class base_Infer(object):
    def __init__(self,model):
        self.model=model
    def sample_pos_W(self,**kwargs):
        raise NotImplementedError('Must override this method!')
    def sample_pos_Z(self,**kwargs):
        raise NotImplementedError('Must override this method!')
    def sample_X(self,**kwargs):
        raise NotImplementedError('Must override this method!')
    def compute_pos_W(self,**kwargs):
        raise NotImplementedError('Must override this method for last layer Gaussian model!')


def train_SGHMC(Infer_model,observed_train,eps=0.01,max_sample_size=20,tot_epoch=100,thinning=50,hyper_param_update=200,sample_int=1,flag_hybrid=False,flag_results=True,W_dict_init=None,**kwargs):
    # This is the function to train the model with SGHMC methods
    if flag_hybrid:
        conditional_coef = kwargs['conditional_coef']
        target_dim = kwargs['target_dim']
    else:
        conditional_coef=1.

    Adam_encoder_PNP_SGHMC=kwargs['Adam_encoder']
    Adam_embedding_PNP_SGHMC=kwargs['Adam_embedding']
    observed_train_size = observed_train.shape[0]
    batch_size = int(observed_train_size / 2.)
    Drop_p=kwargs['Drop_p']
    list_p_z=kwargs['list_p_z']
    test_input_tensor=kwargs['test_input_tensor']
    test_target_tensor=kwargs['test_target_tensor']
    valid_data=kwargs['valid_data']
    valid_data_target=kwargs['valid_data_target']
    W_sample=collections.OrderedDict()
    sigma_out=kwargs['sigma_out']
    scale_data=kwargs['scale_data']
    update_noise=kwargs['noisy_update']
    if W_dict_init is not None:
        W_dict=W_dict_init
    else:
        W_dict=Infer_model.model.decoder._get_W_dict()

    RMSE_mat,MAE_mat,NLL_mat=np.zeros(int(tot_epoch/sample_int)),np.zeros(int(tot_epoch/sample_int)),np.zeros(int(tot_epoch/sample_int))

    if batch_size > 100:
        batch_size = 100
    train_dataset = base_UCI_Dataset(observed_train, transform=None, flag_GPU=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    iter = 1
    Acc_ELBO = 0
    sample_counter=0
    valid_NLL_list=collections.deque(maxlen=10)
    mean_Acc_valid_min=10.
    counter_valid_break=0
    counter=1
    for ep in range(tot_epoch):

        flag_update=True

        for idx, data in enumerate(train_loader):
            Adam_embedding_PNP_SGHMC.zero_grad()
            Adam_encoder_PNP_SGHMC.zero_grad()
            mask = get_mask(data)
            Drop_p_var = np.minimum(np.random.rand(1), Drop_p)
            mask_drop = np.array([bernoulli.rvs(1 - Drop_p_var, size=data.shape[1])] * data.shape[0])
            mask_drop = torch.from_numpy(mask_drop).float().cuda()


            if flag_hybrid:
                mask_drop_hybrid = torch.tensor(mask_drop.data)
                mask_drop_hybrid[:, target_dim] = 1.  # Reserve the target dim
                mask_drop_Xdy = torch.tensor(mask_drop.data)
                mask_drop_Xdy[:, -1] = 0.
                mask_target_ELBO = mask * mask_drop_hybrid
                mask = mask * mask_drop
            else:
                mask =mask*mask_drop#least_one_left_mask(mask,mask_drop)


            # Flag Optimizer (This indicate the burn-in period, if true, it is in burn-in, if false, start sampling)
            flag_optimizer = (ep < (tot_epoch/2))
            if flag_optimizer:
                scale_data=1
            else:
                scale_data=scale_data
            ep_stop=10000
            if  ep<ep_stop:
                flag_record_memory=True
                flag_encoder_update=True
            else:
                # These two flags mean stop update encoder
                flag_record_memory=False
                flag_encoder_update=False
            # Hyper_param Update
            if (ep == 0 or (ep+1) % hyper_param_update == 0) and flag_update==True:
                # Only used for initialize necessary matrix for SGHMC, so only run 1 iter.
                if flag_hybrid:
                    W_dict, r, M, V = Infer_model.sample_pos_W(observed_train, eps=eps, tot_iter=1,
                                                               thinning=200,
                                                               flag_burnin=True,
                                                               flag_reset_r=False, flag_hybrid=flag_hybrid,
                                                               target_dim=target_dim,
                                                               conditional_coef=conditional_coef,W_dict_init=W_dict_init,sigma_out=sigma_out,Drop_p=Drop_p,update_noise=update_noise)
                else:
                    W_dict, r, M, V = Infer_model.sample_pos_W(observed_train, eps=eps, tot_iter=1,
                                                               thinning=200,
                                                               flag_burnin=True,
                                                               flag_reset_r=False,W_dict_init=W_dict_init,sigma_out=sigma_out,Drop_p=Drop_p,update_noise=update_noise)
                flag_update=False
                _,_,r,_,_=Infer_model._init_mat()

            if flag_hybrid:
                ELBO,ELBO_mean=Infer_model._compute_target_ELBO(data, mask, mask_target_ELBO,observed_train_size*scale_data, W_dict, size_Z=10, sigma_out=sigma_out,
                                                 z_sigma_prior=1., W_mean_prior=0., W_sigma_prior=1,
                                                 record_memory=flag_record_memory,target_dim=target_dim,conditional_coef=conditional_coef)
            else:
                ELBO, ELBO_mean = Infer_model._compute_ELBO(data, mask, observed_train_size*scale_data, W_dict, size_Z=10, sigma_out=sigma_out,
                                                 z_sigma_prior=1., W_mean_prior=0., W_sigma_prior=1,
                                                 record_memory=flag_record_memory)



            Acc_ELBO += ELBO_mean.cpu().data.numpy()

            # Obtain grad of ELBO
            (ELBO).backward()
            G_dict = Infer_model.model.decoder._get_grad_W_dict()  # No graph attached




            # SGHMC step
            M_old=copy.deepcopy(M)
            if ep==(tot_epoch/2):
                #V,M,r,_,_=Infer_model._init_mat()
                pass
            W_dict, r,V,M = Infer_model._SGHMC_step(W_dict=W_dict, G_dict=G_dict, eps=eps,eps2=eps, M=M, V=V, r=r,flag_optimizer=flag_optimizer,counter=counter,flag_compensator=True,M_old=M_old,update_noise=update_noise)


            counter+=1
            if flag_encoder_update:
                # Update encoder
                list_grad_p_z = grap_modify_grad(list_p_z, 1, observed_train.shape[0])
                zero_grad(list_p_z)
                assign_grad(list_p_z, list_grad_p_z)
                Adam_embedding_PNP_SGHMC.step()
                Adam_encoder_PNP_SGHMC.step()




            # Initialize W_sample


            iter+=1



        if ep < thinning or flag_optimizer:
            W_sample['sample_1'] = copy.deepcopy(W_dict)
        # Store Samples
        if (ep+1) % thinning == 0 and not flag_optimizer:
            sample_counter += 1
            W_sample = Update_W_sample(W_dict, W_sample, sample_counter, maxsize=max_sample_size)

        if (ep+1)%sample_int==0 and flag_results==True:
            RMSE,MAE,NLL=Test_UCI_batch(Infer_model.model, test_input_tensor, test_target_tensor, sigma_out_scale=sigma_out, flag_fixed=False, split=10,
                           flag_model='PNP_SGHMC', size=10, Infer_model=Infer_model, W_sample=W_sample)
            RMSE_mat[int(ep/sample_int)],MAE_mat[int(ep/sample_int)],NLL_mat[int(ep/sample_int)]=RMSE.cpu().data.numpy(),MAE.cpu().data.numpy(),NLL.cpu().data.numpy()

        # Evaluate the validation NLL
        if (ep + 1) % 10 == 0 and valid_data is not None:
            mask_valid_NLL = get_mask(valid_data)
            mean_valid_NLL, _ = Infer_model.test_log_likelihood(valid_data, valid_data_target,
                                                                W_sample, mask_valid_NLL, sigma_out=sigma_out, size=10)
            valid_NLL_list.append(-mean_valid_NLL.cpu().data.numpy())

            mean_Acc_valid_now = np.mean(valid_NLL_list)
            if (ep+1)>200:
                if mean_Acc_valid_now < mean_Acc_valid_min :
                    mean_Acc_valid_min = mean_Acc_valid_now
                    counter_valid_break = 0
                else:
                    counter_valid_break += 1
            if counter_valid_break > 15:
                RMSE, MAE, NLL = Test_UCI_batch(Infer_model.model, test_input_tensor, test_target_tensor,
                                                sigma_out_scale=sigma_out, flag_fixed=False, split=10,
                                                flag_model='PNP_SGHMC', size=10, Infer_model=Infer_model,
                                                W_sample=W_sample)
                print('validation worse, break at ep:%s with test NLL:%s' % (ep + 1,NLL.cpu().data.numpy()))

                break


        if (ep+1)%100==0:
            RMSE, MAE, NLL = Test_UCI_batch(Infer_model.model, test_input_tensor, test_target_tensor,
                                            sigma_out_scale=sigma_out, flag_fixed=False, split=10,
                                            flag_model='PNP_SGHMC', size=10, Infer_model=Infer_model, W_sample=W_sample)
            # Evaluate the observed train NLL
            mask_train_NLL=get_mask(observed_train)
            mean_train_NLL,_=Infer_model.test_log_likelihood(observed_train, observed_train.clone().detach(), W_sample, mask_train_NLL, sigma_out=sigma_out, size=10)

            # Evaluate the valid NLL
            # Evaluate the observed train NLL
            if valid_data is not None:
                mask_valid_NLL = get_mask(valid_data)
                mean_valid_NLL, _ = Infer_model.test_log_likelihood(valid_data, valid_data_target,
                                                                W_sample, mask_valid_NLL, sigma_out=sigma_out, size=10)
                print('conditional_coef:%s ep:%s train_NLL:%s valid_NLL:%s NLL:%s RMSE:%s'%(conditional_coef,ep+1,mean_train_NLL.cpu().data.numpy(),mean_valid_NLL.cpu().data.numpy(),NLL.cpu().data.numpy(),RMSE.cpu().data.numpy()))
            else:
                print('conditional_coef:%s ep:%s train_NLL:%s NLL:%s RMSE:%s'%(conditional_coef,ep+1,mean_train_NLL.cpu().data.numpy(),NLL.cpu().data.numpy(),RMSE.cpu().data.numpy()))

    return W_sample,RMSE_mat,MAE_mat,NLL_mat





class SGHMC(base_Infer):
    def __init__(self,model,Infer_name='SGHMC'):
        super(SGHMC,self).__init__(model)
        self.Infer_name=Infer_name
        self.flag_only_output_layer=self.model.decoder.flag_only_output_layer
        self.V,self.M,self.r=None,None,None # This is reserved for train_SGHMC() Method
        self.flag_LV=model.flag_LV
    def _SGHMC_step(self,**kwargs):
        W_dict=kwargs['W_dict']
        G_dict=kwargs['G_dict']
        eps=kwargs['eps']
        eps2=kwargs['eps2']
        M=kwargs['M']
        V=kwargs['V']
        r=kwargs['r']
        flag_optimizer=kwargs['flag_optimizer']
        counter=kwargs['counter']
        flag_compensator=kwargs['flag_compensator']
        update_noise=kwargs['update_noise']
        if flag_compensator:
            M_old=kwargs['M_old']
            #W_old=kwargs['W_old']

        # gradient descent
        for key_r,value_r in r.items():
            G_=G_dict[key_r]
            M_=M[key_r] # equivalent to M^-1
            V_=V[key_r]
            W_=W_dict[key_r]
            C_=0.1/(math.sqrt(eps))*torch.sqrt(V_)
            B_=0.5*math.sqrt(eps)*V_
            noise=torch.randn(value_r.shape)
            Diff_CB = torch.clamp((C_ - B_), min=0.)
            #rescaled_noise = 0*torch.sqrt(torch.clamp(2*(eps**3)/(V_)*C_-eps**4,min=0.))*noise  # 1*torch.sqrt(2*eps*Diff_CB) *noise
            # Momentum update
            # value_r.data=value_r.data-eps*G_.data-eps*C_*M_*value_r.data+rescaled_noise # TODO: Check eps*C_*M is 0.05 and (C_-B_) is positive
            # value_r.data = value_r.data - (eps ** 2) / (torch.sqrt(V_)) * G_.data - eps / (
            #     torch.sqrt(V_)) * C_*value_r.data + rescaled_noise
            # position update
            # W_.data=W_.data+eps*M_*value_r.data
            #W_.data = W_.data + value_r.data

            if flag_optimizer:
                # Adam Optimizer (Enable this, must set flag_optimizer to be always True)

                # value_r.data=0.9*value_r.data-0.1*G_.data
                # V_.data=0.99*V_.data+0.01*(G_.data**2)
                # value_r_un=value_r.data/(1-0.9**counter)
                # V_un=V_.data/(1-0.99**counter)
                # W_.data=W_.data+0.003*value_r_un.data/(torch.sqrt(V_un.data))

                V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                M_.data = 1. / torch.sqrt((torch.sqrt(V_.data)))
                C_ = 0.1 / math.sqrt(eps)
                B_ = 0.5 * math.sqrt(eps)
                Diff_CB = C_ - B_
                rescaled_noise = torch.sqrt(2 * Diff_CB * math.sqrt(eps) * eps * M_.data**2) * noise
                # if counter==850:
                #     A=1
                value_r.data = value_r.data - eps * (M_.data ** 2) * G_.data - 0.1 * value_r.data + update_noise*rescaled_noise
                W_.data = W_.data + value_r.data




            else:
                # V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                # M_.data = 1. / (torch.sqrt(V_.data))
                # rescaled_noise=(1*torch.sqrt(2*math.sqrt(eps)*Diff_CB) *noise)/math.sqrt(eps)
                # value_r.data = value_r.data - 0.1 * value_r.data - G_.data+1*rescaled_noise
                # W_.data = W_.data + eps * M_.data * value_r.data

                V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                M_.data = 1. / torch.sqrt((torch.sqrt(V_.data)))
                C_=0.1/math.sqrt(eps2)
                B_ = 0.5 * math.sqrt(eps2)
                Diff_CB=C_-B_



                if flag_compensator:
                    M_old_=M_old[key_r]
                    #W_old_=W_old[key_r]
                    Gamma=(M_.data-M_old_.data)/(torch.sign(value_r.data)*torch.clamp(torch.abs(value_r.data),min=1e-7))
                    #Gamma=torch.clamp(Gamma,min=-10,max=10)
                rescaled_noise=torch.sqrt(2*Diff_CB*math.sqrt(eps2)*eps2*M_.data**2)*noise
                if flag_compensator:
                    value_r.data=value_r.data-eps2*(M_.data**2)*G_.data-0.1*value_r.data+1*rescaled_noise#+1*eps2*M_.data *Gamma.data
                else:
                    value_r.data = value_r.data - eps2 * (
                                M_.data ** 2) * G_.data - 0.1 * value_r.data + 1 * rescaled_noise

                W_.data = W_.data + value_r.data

            # Debug nan detection
            # if torch.sum(torch.isnan(W_.data))>0:
            #     print('nan:%s' %(counter))
            #     raise Exception('nan Occur')




        return W_dict,r,V,M
    def _SGHMC_burnin_step(self,**kwargs):
        # No longer used for burn-in
        W_dict = kwargs['W_dict']
        G_dict = kwargs['G_dict']
        eps = kwargs['eps']
        M = kwargs['M']
        V = kwargs['V']
        r = kwargs['r']
        tau=kwargs['tau']
        flag_init=kwargs['flag_init']
        flag_update_tau=kwargs['flag_update_tau']
        #flag_optimizer=kwargs['flag_optimizer']
        if flag_init:
            g = copy.deepcopy(G_dict)
            V=copy.deepcopy(square_dict(G_dict))

        else:
            g=kwargs['g']
        # Update hyperparameter
        for key_r, value_r in r.items():
            g_=g[key_r]
            tau_=tau[key_r]
            G_=G_dict[key_r]
            V_=V[key_r]
            # Update g and V
            g_.data=g_.data-1./tau_*g_.data+1./tau_*G_.data
            V_.data=torch.clamp(V_.data-1./tau_*V_.data+1./(tau_)*((G_.data)**2),min=0.0001)
            if flag_update_tau:
                # Update window
                tau_.data=tau_.data-((g_.data)**2)/((V_.data))*tau_.data+1
                tau_.data=torch.clamp(tau_.data,min=1.01) # set minimum window size
                tau_.data.fill_(1.01)

        # Update the model
        for key_r,value_r in r.items():
            G_=G_dict[key_r]
            V_=V[key_r]
            M_ = 1. / torch.sqrt(V_)  # equivalent to M^-1
            M[key_r] = M_
            W_=W_dict[key_r]
            C_=0.1/(eps)*torch.sqrt(V_)
            B_=0.5*eps*V_
            noise=torch.randn(value_r.shape)
            Diff_CB=torch.clamp((C_-B_),min=0.)
            rescaled_noise=0*torch.sqrt(torch.clamp(2*(eps**3)/(V_)*C_-eps**4,min=0.))*noise#1*torch.sqrt(2*eps*Diff_CB) *noise

            value_r.data=value_r.data-0.1*value_r.data-G_.data
            W_.data=W_.data+0.003*M_.data*value_r.data

        return W_dict,r,M,V,tau,g


    def _init_mat(self):
        # This is used to initialize the M and V and r
        W_dict=self.model.decoder._get_W_dict()
        V=collections.OrderedDict()
        M=collections.OrderedDict()
        r=collections.OrderedDict()
        tau=collections.OrderedDict()
        g=collections.OrderedDict()
        if not self.flag_only_output_layer:
            for layer_ind in range(self.model.decoder.hidden_layer_num):
                V['weight_layer_%s'%(layer_ind)]=torch.ones(W_dict['weight_layer_%s'%(layer_ind)].shape)
                V['bias_layer_%s'%(layer_ind)]=torch.ones(W_dict['bias_layer_%s'%(layer_ind)].shape)
                M['weight_layer_%s'%(layer_ind)] = 1. / torch.sqrt(V['weight_layer_%s'%(layer_ind)])
                M['bias_layer_%s'%(layer_ind)] = 1. / (torch.sqrt(V['bias_layer_%s'%(layer_ind)]))
                r['weight_layer_%s'%(layer_ind)]=0.01*torch.randn(W_dict['weight_layer_%s'%(layer_ind)].shape)
                r['bias_layer_%s'%(layer_ind)]=0.01*torch.randn(W_dict['bias_layer_%s'%(layer_ind)].shape)
                tau['weight_layer_%s' % (layer_ind)] = 1.01 * torch.ones(W_dict['weight_layer_%s' % (layer_ind)].shape)
                tau['bias_layer_%s' % (layer_ind)] = 1.01 * torch.ones(W_dict['bias_layer_%s' % (layer_ind)].shape)
        # Output layer
        V['weight_out'] = torch.ones(W_dict['weight_out'].shape)
        V['bias_out'] = torch.ones(W_dict['bias_out'].shape)
        M['weight_out'] = 1. / torch.sqrt(V['weight_out'])
        M['bias_out'] = 1. / (torch.sqrt(V['bias_out']))
        r['weight_out'] = 0.01 * torch.randn(W_dict['weight_out'].shape)
        r['bias_out'] = 0.01 * torch.randn(W_dict['bias_out'].shape)
        tau['weight_out'] = 1.01 * torch.ones(W_dict['weight_out'].shape)
        tau['bias_out'] = 1.01 * torch.ones(W_dict['bias_out'].shape)
        if self.flag_LV:
            V['weight_out_LV'] = torch.ones(W_dict['weight_out_LV'].shape)
            V['bias_out_LV'] = torch.ones(W_dict['bias_out_LV'].shape)
            M['weight_out_LV'] = 1. / torch.sqrt(V['weight_out_LV'])
            M['bias_out_LV'] = 1. / (torch.sqrt(V['bias_out_LV']))
            r['weight_out_LV'] = 0.01 * torch.randn(W_dict['weight_out_LV'].shape)
            r['bias_out_LV'] = 0.01 * torch.randn(W_dict['bias_out_LV'].shape)
            tau['weight_out_LV'] = 1.01 * torch.ones(W_dict['weight_out_LV'].shape)
            tau['bias_out_LV'] = 1.01 * torch.ones(W_dict['bias_out_LV'].shape)

        g=copy.deepcopy(tau)


        return V,M,r,tau,g

    def sample_pos_W(self,observed_train,eps=0.01,tot_iter=500,thinning=50,flag_burnin=False,flag_reset_r=False,flag_hybrid=False,W_dict_init=None,**kwargs):
        # this is to draw samples of W (no longer used now, sample W is drawn using train_SGHMC method), this is only used as hyperparameter initialization
        if flag_hybrid:
            conditional_coef=kwargs['conditional_coef']
            target_dim=kwargs['target_dim']
        if flag_burnin:
            if W_dict_init is not None:
                W_dict=W_dict_init
            else:
                W_dict=self.model.decoder._get_W_dict() # No graph attached
        else:
            W_dict=kwargs['W_dict']
        W_sample=collections.OrderedDict()
        Drop_p=kwargs['Drop_p']
        if flag_burnin==True:
            V,M,r,tau,g=self._init_mat()
        else:
            M = kwargs['M']
            V = kwargs['V']
            r = kwargs['r']

        if flag_reset_r:
            _,_,r,_,_=self._init_mat()

        observed_train_size=observed_train.shape[0] # Assume the observed_train is after removing 0 row
        batch_size=int(observed_train_size/2.)
        if batch_size>100:
            batch_size=100
       # Define Data loader
        train_dataset = base_UCI_Dataset(observed_train, transform=None, flag_GPU=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        sigma_out=kwargs['sigma_out']
        iter=0
        Acc_ELBO=0
        sample_counter=0
        update_noise=kwargs['update_noise']
        while iter<=tot_iter:
            for idx, data in enumerate(train_loader):
                mask = get_mask(data)
                Drop_p_var = np.minimum(np.random.rand(1), Drop_p)
                mask_drop = np.array([bernoulli.rvs(1 - Drop_p_var, size=data.shape[1])] * data.shape[0])
                mask_drop = torch.from_numpy(mask_drop).float().cuda()

                if flag_hybrid:
                    mask_drop_hybrid = torch.tensor(mask_drop.data)
                    mask_drop_hybrid[:, target_dim] = 1.  # Reserve the target dim
                    mask_drop_Xdy = torch.tensor(mask_drop.data)
                    mask_drop_Xdy[:, -1] = 0.
                    mask_target_ELBO = mask * mask_drop_hybrid
                    mask = mask * mask_drop
                else:
                    mask = mask * mask_drop


                if (iter)%thinning==0 and not flag_burnin and iter>0:
                    flag_record=True
                else:
                    flag_record=False
                # Compute ELBO
                if flag_hybrid:
                    ELBO,ELBO_mean=self._compute_target_ELBO(data,mask,mask_target_ELBO,observed_train_size,W_dict,size_Z=10,sigma_out=sigma_out,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=flag_record,target_dim=target_dim,conditional_coef=conditional_coef)
                else:
                    ELBO,ELBO_mean=self._compute_ELBO(data,mask,observed_train_size,W_dict,size_Z=10,sigma_out=sigma_out,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=flag_record)


                Acc_ELBO +=ELBO_mean.cpu().data.numpy()


                # Obtain grad of ELBO
                (ELBO).backward()
                G_dict=self.model.decoder._get_grad_W_dict() # No graph attached

                # SGHMC update
                if not flag_burnin:
                    W_dict,r=self._SGHMC_step(W_dict=W_dict,G_dict=G_dict,eps=eps,M=M,V=V,r=r,update_noise=update_noise)
                else:
                    W_dict, r, M, V, tau, g=self._SGHMC_burnin_step(W_dict=W_dict,G_dict=G_dict,eps=eps,M=M,V=V,r=r,tau=tau,g=g,flag_init=(iter==0),flag_update_tau=(iter>50))


                # Update iter
                iter+=1
                if (iter)%thinning==0:
                    sample_counter+=1
                    # Store samples
                    if not flag_burnin:
                        W_sample['sample_%s'%(sample_counter)]=copy.deepcopy(W_dict)
                    #print('Iter:%s Acc_ELBO:%s'%(iter,Acc_ELBO/thinning))
                    Acc_ELBO=0.
                if iter>tot_iter:
                    break

        if flag_burnin:
            r,M,V=copy.deepcopy(r),copy.deepcopy(M),copy.deepcopy(V)
            return W_dict,r,M,V
        else:
            return W_sample,W_dict

    # def burn_in(self,W_dict,observed_train,eps=0.01,tot_iter=500,thinning=50,**kwargs):
    #     W_dict = self.model.decoder._get_W_dict()  # No graph attached
    #     W_sample = {}
    #     V, M, r,tau,g = self._init_mat()


    def sample_pos_Z(self,X,mask,size_Z=10):
        z, encoding = self.model.sample_latent_variable(X, mask, size=size_Z) # N_z x N x latent_dim
        return z,encoding

    def completion(self,X,mask,W_sample,size_Z=10,record_memory=False):
        # Imputing all missing values
        X = X.clone().detach()
        with torch.no_grad():
            z, _ = self.model.sample_latent_variable(X, mask, size=size_Z)
        if record_memory==False:
            z=z.clone().detach()
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            X=self.sample_X(z,W_sample) # N_w x N_z x N x obs_dim
            del z
            return X

    def sample_X(self,z,W_sample):
        # z is N_z x N x latent
        counter=0
        for key_W,value_W in W_sample.items():
            self.model.decoder._assign_weight(value_W)
            # Compute the log likelihood
            # Decode
            decode = self.model.decoder.forward(z) # N_z x N x obs_dim
            # Remove memory
            decode=decode.clone().detach()
            if self.flag_LV:
                raise RuntimeError('flag_LV must be False')


            else:
                if counter==0:

                    decode=torch.unsqueeze(decode,dim=0) # 1 x N_z x N x obs_dim
                    X=decode
                else:
                    decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                    X = torch.cat((X,decode),dim=0) # N_W x N_z x N x obs_dim
                counter+=1

        return X


    def _compute_ELBO(self,X,mask,data_N,W_dict,size_Z=10,sigma_out=0.1,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=False):
        num_X=X.shape[0]
        obs_dim=X.shape[1]
        # Sample Z_i
        z,encoding=self.model.sample_latent_variable(X,mask,size=size_Z) # N_z x N x latent_dim

        q_mean = encoding[:, :self.model.latent_dim]

        # q_sigma=F.softplus(encoding[:,self.latent_dim:]) # N x latent_dim
        if self.model.flag_log_q == True:
            q_sigma = torch.sqrt(torch.exp(encoding[:, self.model.latent_dim:]))
        else:
            q_sigma = torch.clamp(torch.sqrt(torch.clamp((encoding[:, self.model.latent_dim:]) ** 2,min=1e-8)),min=1e-4,max=30)

        if not record_memory:
            # Remove memory
            z=z.clone().detach()
            q_mean,q_sigma=q_mean.clone().detach(),q_sigma.clone().detach()

        #  No need to compute the KL term because no gradient w.r.t W
        # Compute the log likelihood term

        # Assign W_dict
        self.model.decoder._assign_weight(W_dict)
        # Compute the log likelihood
        # Decode
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            decode_X=self.model.decoder.forward(z) # Nz x N x obs_dim

            # Log likelihood
            log_likelihood=self._scaled_gaussian_log_likelihood(X, mask, decode_X, sigma_out,sigma_out_target_scale=1.) # N_z x N
            ELBO=log_likelihood-1.*torch.unsqueeze(self.model._KL_encoder(q_mean,q_sigma,z_sigma_prior),dim=0) # N_z x N
            ELBO=torch.sum(torch.mean(ELBO,dim=0) )/num_X*data_N+1*self._compute_W_prior(W_mean_prior=W_mean_prior,W_sigma_prior=W_sigma_prior) # TODO: Need to check if we only have prior, does it have gradient? Yes!
            ELBO_mean=ELBO/data_N
        return ELBO,ELBO_mean
    def _compute_target_ELBO(self,X,mask,mask_target_ELBO,data_N,W_dict,size_Z=10,sigma_out=0.1,z_sigma_prior=1.,W_mean_prior=0.,W_sigma_prior=1.,record_memory=False,target_dim=-1,conditional_coef=0.5):
        # ASSUME THE X CONTAIN THE TARGET VARIABLE and MASK IS FOR XUY!!!!! mask is for reconstruction mask_target_ELBO is for target loss


        num_X = X.shape[0]
        obs_dim=X.shape[1]
        XUy = torch.tensor(X.data)  # with target variable
        Xdy = torch.tensor(X.data)
        X=X.clone().detach()

        Xdy[:, target_dim] = 0.  # zero the target dim
        mask_Xdy = torch.tensor(mask_target_ELBO.data)
        mask_Xdy[:, target_dim] = 0.

        z_XUy, encoding_XUy = self.model.sample_latent_variable(XUy, mask_target_ELBO, size=size_Z)  # N_z x N x latent_dim
        z_Xdy, encoding_Xdy = self.model.sample_latent_variable(Xdy, mask_Xdy, size=size_Z)  # N_z x N x latent_dim

        z_X,encoding_X = self.model.sample_latent_variable(X, mask, size=size_Z)  # N_z x N x latent_dim

        q_mean_XUy = encoding_XUy[:, :self.model.latent_dim]
        q_mean_Xdy = encoding_Xdy[:, :self.model.latent_dim]
        q_mean_X = encoding_X[:, :self.model.latent_dim]

        # q_sigma=F.softplus(encoding[:,self.latent_dim:]) # N x latent_dim
        if self.model.flag_log_q == True:
            q_sigma_XUy = torch.sqrt(torch.exp(encoding_XUy[:, self.model.latent_dim:]))
            q_sigma_Xdy = torch.sqrt(torch.exp(encoding_Xdy[:, self.model.latent_dim:]))
            q_sigma_X = torch.sqrt(torch.exp(encoding_X[:, self.model.latent_dim:]))

        else:
            q_sigma_XUy = torch.clamp(torch.sqrt((encoding_XUy[:, self.model.latent_dim:]) ** 2), min=1e-5, max=30.)
            q_sigma_Xdy = torch.clamp(torch.sqrt((encoding_Xdy[:, self.model.latent_dim:]) ** 2), min=1e-5, max=30.)
            q_sigma_X = torch.clamp(torch.sqrt((encoding_X[:, self.model.latent_dim:]) ** 2), min=1e-5, max=30.)

        if not record_memory:
            # Remove memory
            z_XUy=z_XUy.clone().detach()
            z_Xdy = z_Xdy.clone().detach()
            z_X=z_X.clone().detach()

            q_mean_XUy,q_sigma_XUy,q_sigma_X=q_mean_XUy.clone().detach(),q_sigma_XUy.clone().detach(),q_sigma_X.clone().detach()
            q_mean_Xdy, q_sigma_Xdy,q_mean_X = q_mean_Xdy.clone().detach(), q_sigma_Xdy.clone().detach(),q_mean_X.clone().detach()

        q_mean_Xdy=q_mean_Xdy.clone().detach()
        q_sigma_Xdy=q_sigma_Xdy.clone().detach()


        KL_z_target = self.model._KL_encoder_target_ELBO(q_mean_XUy, q_sigma_XUy, q_mean_Xdy, q_sigma_Xdy)  # N
        KL_z=self.model._KL_encoder(q_mean_X,q_sigma_X,z_sigma_prior) # N

        # Assign W_dict
        self.model.decoder._assign_weight(W_dict)

        # Reconstruction loss
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:

            decode_X = self.model.decoder.forward(z_X)  # Nz x N x obs_dim
            # Log likelihood
            log_likelihood_recon = self._scaled_gaussian_log_likelihood(X, mask, decode_X, sigma_out,flag_target=False)  # N_z x N
            ELBO_recon = 1*log_likelihood_recon - 1*torch.unsqueeze(KL_z,
                                                    dim=0)  # N_z x N
            ELBO_recon = torch.sum(torch.mean(ELBO_recon, dim=0))/ num_X * data_N  + 1*self._compute_W_prior(W_mean_prior=W_mean_prior,
                                                                                               W_sigma_prior=W_sigma_prior)  # TODO: Need to check if we only have prior, does it have gradient? Yes!
            ELBO_mean_recon = ELBO_recon / data_N

        # Target loss
        mask_y = torch.zeros(mask.shape)
        mask_y[:, target_dim] = 1.  # Only reserve the target dim and disable all other

        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:

            decode_target = self.model.decoder.forward(z_XUy)  # Nz x N x obs_dim

            log_likelihood_target = self._scaled_gaussian_log_likelihood(XUy, mask_y, decode_target, sigma_out,flag_target=True)  # N_z x N
            ELBO_target = log_likelihood_target - 1.*torch.unsqueeze(KL_z_target,
                                                    dim=0)  # N_z x N
            ELBO_target = torch.sum(torch.mean(ELBO_target, dim=0)) / num_X * data_N + 1*self._compute_W_prior(W_mean_prior=W_mean_prior,
                                                                                               W_sigma_prior=W_sigma_prior)
            ELBO_mean_target = ELBO_target / data_N

        # Hybrid model
        ELBO=conditional_coef*ELBO_recon+(1-conditional_coef)*ELBO_target
        ELBO_mean=ELBO_target
        return ELBO, ELBO_mean
    def _scaled_gaussian_log_likelihood(self,X,mask,decode,sigma_out,sigma_out_target_scale=1.,flag_target=False,**kwargs):
        if self.flag_LV:
            decode_LV=kwargs['decode_X_LV']
        # decode is Nz x N x obs_dim
        active_dim_scale=torch.clamp(torch.sum(torch.abs(get_mask(X))>0.,dim=1).float(),min=1) # N
        X = X * mask
        decoding_size = len(decode.shape)
        obs_dim=X.shape[1]
        if decoding_size==3: # Nz x N x obs_dim
            X_expand = torch.unsqueeze(X, dim=0)
            mask_expand = torch.unsqueeze(mask, dim=0)  # 1 x N x obs_dim

            if flag_target==False:
                active_dim=torch.clamp(torch.sum(torch.abs(mask)>0.,dim=1).float(),min=1) # N
            else:
                active_dim=1.

            decoding = decode * mask_expand  # N_z x N x obs_dim
            if self.flag_LV:
                raise RuntimeError('flag_LV must be False')
            else:
                # Apart from target
                X_expand_X=X_expand[:,:,0:-1]
                X_expand_Y=torch.unsqueeze(X_expand[:,:,-1],dim=2) # N_z x N x 1
                decoding_X=decoding[:,:,0:-1]
                decoding_Y=torch.unsqueeze(decoding[:,:,-1],dim=2) # N_z x N x 1
                log_likelihood= -0.5 * torch.sum((X_expand_X - decoding_X) ** 2 / (sigma_out ** 2),
                                                  dim=2)-0.5*torch.sum((X_expand_Y-decoding_Y)**2/((sigma_out*sigma_out_target_scale)**2),dim=2) - 0.5 * torch.unsqueeze(torch.sum(mask, dim=1), dim=0) * (
                                             math.log(sigma_out ** 2) + math.log(2 * np.pi)) # N_z x N
                # log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                #                                   dim=2) - 0.5 * torch.unsqueeze(torch.sum(mask, dim=1), dim=0) * (
                #                              math.log(sigma_out ** 2) + math.log(2 * np.pi)) # N_z x N
            #


            # Scale it with |D/D_0|
            if flag_target==False:
                active_dim_scale_expand = torch.unsqueeze(active_dim_scale, dim=0)
                active_dim_expand=torch.unsqueeze(active_dim,dim=0) # 1 x N

                log_likelihood=log_likelihood/active_dim_expand*active_dim_scale_expand # N_z x N
            else:
                active_dim_scale_expand = torch.unsqueeze(active_dim_scale, dim=0)
                log_likelihood = log_likelihood#*obs_dim/active_dim_scale_expand #TODO: This is to maintain the relative scale of conditional model and reconstruction model

        else:
            raise NotImplementedError
        return log_likelihood

    def _compute_W_prior(self,W_mean_prior=0.,W_sigma_prior=1.):
        flat_weight,out_wight,out_bias=self.model.decoder._flatten_stat() # 1 x D_weight,N_in x N_out and N_out
        if self.model.decoder.flag_only_output_layer==False:
            # Prior of flat_weight
            log_prior_shared=torch.sum(-0.5/(W_sigma_prior**2)*((flat_weight-W_mean_prior)**2)-0.5*math.log(W_sigma_prior**2)-0.5*math.log(2*np.pi))
            # Compute out_weight prior
            log_prior_out_weight=torch.sum(-0.5/(W_sigma_prior**2)*((out_wight-W_mean_prior)**2)-0.5*math.log(W_sigma_prior**2)-0.5*math.log(2*np.pi))
            log_prior_out_bias=torch.sum(-0.5/(W_sigma_prior**2)*((out_bias-W_mean_prior)**2)-0.5*math.log(W_sigma_prior**2)-0.5*math.log(2*np.pi))
            log_prior=log_prior_shared+log_prior_out_weight+log_prior_out_bias
        else:
            # Compute out_weight prior
            log_prior_out_weight = torch.sum(
                -0.5 / (W_sigma_prior ** 2) * ((out_wight - W_mean_prior) ** 2) - 0.5 * math.log(
                    W_sigma_prior ** 2) - 0.5 * math.log(2 * np.pi))
            log_prior_out_bias = torch.sum(
                -0.5 / (W_sigma_prior ** 2) * ((out_bias - W_mean_prior) ** 2) - 0.5 * math.log(
                    W_sigma_prior ** 2) - 0.5 * math.log(2 * np.pi))
            log_prior = log_prior_out_weight + log_prior_out_bias

        return log_prior
    def test_log_likelihood(self,X_in,X_test,W_sample,mask,sigma_out,size=10):
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            with torch.no_grad():
                complete=self.completion(X_in,mask,W_sample,size_Z=size,record_memory=False)
            complete=complete.clone().detach()
            test_mask=get_mask(X_test)

            X_expand = torch.unsqueeze(torch.unsqueeze(X_test, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            mask_expand = torch.unsqueeze(torch.unsqueeze(test_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            decoding = complete * mask_expand  # N_w x N_z x N x obs_dim
            with torch.no_grad():
            # Proper NLL
                log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                                              dim=3) - 0.5 * torch.unsqueeze(
                torch.unsqueeze(torch.sum(test_mask, dim=1), dim=0),
                dim=0) * (math.log(
                sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N_z x N


                log_likelihood = log_likelihood.view(-1, log_likelihood.shape[2])  # (N_w x N_z) x N


                pred_log_likelihood = torch.logsumexp(log_likelihood, dim=0) - math.log(float(log_likelihood.shape[0]))  # N


            # Expected log (ELBO)

            # log_likelihood=-0.5*torch.sum((X_expand-decoding)**2/(sigma_out**2),dim=3)-math.log(sigma_out) # N_W x N_z x N
            # log_likelihood = log_likelihood.view(-1, log_likelihood.shape[2])  # (N_w x N_z) x N
            # pred_log_likelihood = torch.mean(log_likelihood, dim=0)


                mean_pred_log_likelihood = 1. / (torch.sum(test_mask)) * torch.sum(pred_log_likelihood)
                tot_pred_ll = torch.sum(pred_log_likelihood)
            del complete
            del W_sample
        return torch.tensor(mean_pred_log_likelihood.data), torch.tensor(tot_pred_ll.data)  # Clear Memory
