import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from base_Model.base_Network import *
from PA_BELGAM.PA_BELGAM_model import *
from Util.Util_func import *
from torch.utils.data import DataLoader
from scipy.stats import bernoulli
from sklearn.utils import shuffle
from torch import autograd
from PA_BELGAM.PA_BELGAM_Dataloader import *
from torchvision import datasets,transforms
import torchvision
import matplotlib.pyplot as plt

class SGHMC(object):
    def __init__(self,model):
        self.model=model

        self.flag_only_output_layer = self.model.decoder.flag_only_output_layer
    def _SGHMC_step(self,**kwargs):
        flag_adam=kwargs['flag_adam']
        W_dict=kwargs['W_dict']
        G_dict=kwargs['G_dict']
        eps=kwargs['eps']
        eps2 = kwargs['eps2']
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
                # Adam version
                # V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                # M_.data = 1. / (torch.sqrt(V_.data))
                # rescaled_noise = (1 * torch.sqrt(2 * math.sqrt(eps) * Diff_CB) * noise) / math.sqrt(eps)
                # value_r.data = value_r.data - 0.1 * value_r.data - G_.data + 0 * rescaled_noise
                # W_.data = W_.data + eps * M_.data * value_r.data
                if flag_adam:
                    value_r.data=0.9*value_r.data-0.1*G_.data
                    V_.data=0.99*V_.data+0.01*(G_.data**2)
                    value_r_un=value_r.data/(1-0.9**counter)
                    V_un=V_.data/(1-0.99**counter)
                    W_.data=W_.data+0.002*value_r_un.data/(torch.sqrt(V_un.data))
                else:
                    V_.data = 0.99 * V_.data + 0.01 * (G_.data ** 2)
                    M_.data = 1. / torch.sqrt((torch.sqrt(V_.data)))
                    C_ = 0.1 / math.sqrt(eps)
                    B_ = 0.5 * math.sqrt(eps)
                    Diff_CB = C_ - B_
                    rescaled_noise = torch.sqrt(2 * Diff_CB * math.sqrt(eps) * eps * M_.data**2) * noise
                    # if counter==1750:
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




        return W_dict,r,V,M # TODO: Check if the value is indeed updated
    def train_SGHMC(self,overall_x,eps=0.0001,max_sample_size=40,tot_epoch=800,thinning=20,result_interval=1,flag_results=True,**kwargs):
        batch_size = int(overall_x.shape[0] / 2.)
        flag_adam=kwargs['flag_adam']



        # Pass Optimizer
        Adam_encoder=kwargs['Adam_encoder']

        # list of trainable parameters
        list_p_z = kwargs['list_p_z']
        # test data
        overall_test=kwargs['test_data']
        W_sample = collections.OrderedDict()
        sigma_out = kwargs['sigma_out']
        W_dict = self.model.decoder._get_W_dict()
        if flag_results==True:
            RMSE_mat, MAE_mat, NLL_mat = np.zeros(int(tot_epoch / result_interval)), np.zeros(
            int(tot_epoch / result_interval)), np.zeros(int(tot_epoch / result_interval))
        if batch_size > 100:
            batch_size = 100
        # dataset and data loader
        UCI_dataset=base_UCI_Dataset_PABELGAM(overall_x,transform=None,flag_GPU=True)
        UCI_train_loader=DataLoader(UCI_dataset,batch_size=batch_size,shuffle=True)
        data_N=overall_x.shape[0]
        counter=1
        sample_counter=0
        result_counter=0
        # training loop
        for ep in range(tot_epoch):

            if flag_adam == True:
                flag_optimizer=True
                flag_prior=False
            else:
                flag_optimizer = (ep < (tot_epoch / 2))
                flag_prior=True

            if ep<tot_epoch-100:
                flag_encoder_update=True
                flag_record_memory=True
            else:
                flag_encoder_update = False
                flag_record_memory = False

            Acc_ELBO=0
            for idx,data in enumerate(UCI_train_loader):
                Adam_encoder.zero_grad()
                x,y=data

                # initialize the SGHMC
                if ep==0:
                    V,M,r,tau,g=self._init_mat()
                    W_dict = self.model.decoder._get_W_dict()
                # ELBO
                ELBO=self._compute_ELBO(x,y,data_N,W_dict,sigma_out=sigma_out,W_mean_prior=0.,W_sigma_prior=1.,record_memory=flag_record_memory,flag_prior=flag_prior)
                Acc_ELBO+=ELBO
                # Backwards
                ELBO.backward()


                G_dict = self.model.decoder._get_grad_W_dict()

                # SGHMC step

                W_dict,r,V,M=self._SGHMC_step(W_dict=W_dict, G_dict=G_dict, eps=eps,eps2=eps, M=M, V=V, r=r,flag_optimizer=flag_optimizer,counter=counter,flag_compensator=False,update_noise=0,flag_adam=flag_adam)
                counter+=1

                if flag_encoder_update:
                    list_grad_p_z = grap_modify_grad(list_p_z, 1, overall_x.shape[0])
                    zero_grad(list_p_z)
                    assign_grad(list_p_z, list_grad_p_z)
                    Adam_encoder.step()

            # Training progress:


            if ep < thinning or flag_optimizer:
                W_sample['sample_1'] = copy.deepcopy(W_dict)

            if (ep + 1) % thinning == 0 and not flag_optimizer:
                sample_counter += 1
                W_sample = Update_W_sample(W_dict, W_sample, sample_counter, maxsize=max_sample_size)



            if flag_results==True and (ep+1)%result_interval==0:
                print('Training: ep:%s ELBO:%s' % (ep, (Acc_ELBO / data_N).cpu().data.numpy()))
                RMSE,MAE,NLL=Test_UCI(self.model,self.test_log_likelihood,W_sample,overall_test,sigma_out=sigma_out,split=3)

                print('Test: ep:%s RMSE:%s MAE:%s NLL:%s'%(ep,RMSE.cpu().data.numpy(),MAE.cpu().data.numpy(),NLL.cpu().data.numpy()))
                RMSE_mat[result_counter],MAE_mat[result_counter],NLL_mat[result_counter]=RMSE.cpu().data.numpy(),MAE.cpu().data.numpy(),NLL.cpu().data.numpy()
                result_counter+=1
        if flag_results:
            return RMSE_mat,MAE_mat,NLL_mat










    def test_log_likelihood(self,x_test,y_test,W_sample,sigma_out):
        with torch.no_grad():
            completion=self.model.completion(x_test,W_sample)
        completion = completion.clone().detach() # Nw x Nz x N x out
        y_expand = torch.unsqueeze(torch.unsqueeze(y_test, dim=0), dim=0).repeat(completion.shape[0], completion.shape[1],
                                                                                 1, 1)  # Nw x Nz x N x obs_dim

        with torch.no_grad():
            log_likelihood = -0.5 * torch.sum((y_expand - completion) ** 2 / (sigma_out ** 2),
                                              dim=3) - 0.5 * completion.shape[-1] * (math.log(
                sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N_z x N

            log_likelihood=log_likelihood.view(-1, log_likelihood.shape[2]) #(nw x nz) x N
            pred_log_likelihood = torch.logsumexp(log_likelihood, dim=0) - math.log(float(log_likelihood.shape[0]))  # N

            mean_pred_log_likelihood=torch.mean(pred_log_likelihood)
            tot_log_likelihood=torch.sum(pred_log_likelihood)

        return tot_log_likelihood





    def _init_mat(self):
        # This is used to initialize the M and V and r
        W_dict = self.model.decoder._get_W_dict()
        V = collections.OrderedDict()
        M = collections.OrderedDict()
        r = collections.OrderedDict()
        tau = collections.OrderedDict()
        g = collections.OrderedDict()
        if not self.flag_only_output_layer:
            for layer_ind in range(self.model.decoder.hidden_layer_num):
                V['weight_layer_%s' % (layer_ind)] = torch.ones(W_dict['weight_layer_%s' % (layer_ind)].shape)
                V['bias_layer_%s' % (layer_ind)] = torch.ones(W_dict['bias_layer_%s' % (layer_ind)].shape)
                M['weight_layer_%s' % (layer_ind)] = 1. / torch.sqrt(V['weight_layer_%s' % (layer_ind)])
                M['bias_layer_%s' % (layer_ind)] = 1. / (torch.sqrt(V['bias_layer_%s' % (layer_ind)]))
                r['weight_layer_%s' % (layer_ind)] = 0.01 * torch.randn(W_dict['weight_layer_%s' % (layer_ind)].shape)
                r['bias_layer_%s' % (layer_ind)] = 0.01 * torch.randn(W_dict['bias_layer_%s' % (layer_ind)].shape)
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
        return V, M, r, tau, g
    def _compute_ELBO(self,x,y,data_N,W_dict,sigma_out=0.1,W_mean_prior=0.,W_sigma_prior=1.,record_memory=False,flag_prior=True):
        if flag_prior==False:
            W_const=0.
        else:
            W_const=1.
        num_X = x.shape[0]
        output_dim = y.shape[1]
        z_recog,mean_recog,sigma_recog=self.model.sample_latent_variable(x, y, flag_prior_net=False)
        _,mean_prior,sigma_prior=self.model.sample_latent_variable(x,y=None,flag_prior_net=True)
        hybrid_coef=self.model.hybrid_coef
        if not record_memory:
            z_recog = z_recog.clone().detach()
            mean_recog,sigma_recog,mean_prior,sigma_prior=mean_recog.clone().detach(),sigma_recog.clone().detach()\
                ,mean_prior.clone().detach(),sigma_prior.clone().detach()

        #assign the W_dict
        self.model.decoder._assign_weight(W_dict)

        decode_y = self.model.decoding(x,z_recog) # Nz x N x out

        log_likelihood=self._gaussian_log_likelihood(y,decode_y,sigma_out=sigma_out) # Nz x N

        ELBO_CVAE=log_likelihood-self.model.KL_coef*torch.unsqueeze(self.model._KL_encoder(mean_prior,sigma_prior,mean_recog,sigma_recog),dim=0)
        ELBO_CVAE=torch.sum(torch.mean(ELBO_CVAE,dim=0))/num_X*data_N+W_const*self._compute_W_prior(W_mean_prior=W_mean_prior,W_sigma_prior=W_sigma_prior)

        # ELBO_GSSN
        z_prior, _,_ = self.model.sample_latent_variable(x, y=None, flag_prior_net=True)
        if not record_memory:
            z_prior = z_prior.clone().detach()

        decode_y_GSSN = self.model.decoding(x, z_prior)
        log_likelihood_GSSN = self._gaussian_log_likelihood(y, decode_y_GSSN, sigma_out=sigma_out)  # Nz x N
        ELBO_GSSN=torch.sum(torch.mean(log_likelihood_GSSN,dim=0))/num_X*data_N+W_const*self._compute_W_prior(W_mean_prior=W_mean_prior,W_sigma_prior=W_sigma_prior)
        ELBO=self.model.hybrid_coef*ELBO_CVAE+(1-self.model.hybrid_coef)*ELBO_GSSN

        return ELBO

    def _gaussian_log_likelihood(self,y,decode,sigma_out=0.1):
        # decode is Nz x N x out
        y_expand = torch.unsqueeze(y, dim=0).repeat(decode.shape[0], 1, 1) # Nz x N x out

        log_likelihood = -0.5 * torch.sum((y_expand - decode) ** 2 / (sigma_out ** 2),
                                          dim=2)- 0.5 * self.model.output_dim * (
                                             math.log(sigma_out ** 2) + math.log(2 * np.pi)) # N_z x N

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


def show_batch(batch,idx=0,flag_recons=False):
    if flag_recons==True:
        batch=batch[0:1,:,:].view(-1,784)
    else:
        batch=batch[idx:idx+1,:]
    batch=batch.view(-1,1,28,28)
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)))
    plt.savefig('/home/paperspace/Projects/SEDDI/EDDI_Experiment/PA_BELGAM/MNIST_test.jpg')


class SGHMC_MNIST(SGHMC):
    def __init__(self,*args,**kwargs):
        super(SGHMC_MNIST,self).__init__(*args,**kwargs)



    def train_SGHMC(self,eps=0.0001,max_sample_size=40,tot_epoch=30,thinning=20,result_interval=10,flag_results=True,**kwargs):
        flag_adam = kwargs['flag_adam']

        # Pass Optimizer
        Adam_encoder = kwargs['Adam_encoder']

        # list of trainable parameters
        list_p_z = kwargs['list_p_z']
        W_sample = collections.OrderedDict()
        sigma_out = kwargs['sigma_out']
        W_dict = self.model.decoder._get_W_dict()

        # dataset and data loader
        train_loader=torch.utils.data.DataLoader(datasets.MNIST('../data',train=True,download=True,transform=transforms.ToTensor()),
                                                 batch_size=100,shuffle=True)
        test_loader=torch.utils.data.DataLoader(datasets.MNIST('../data',train=False,download=True,transform=transforms.ToTensor()),
                                                 batch_size=100,shuffle=True)
        data_N = len(datasets.MNIST('../data',train=True,download=True,transform=transforms.ToTensor()))
        counter = 1
        iter_counter=0
        sample_counter = 0
        result_counter = 0
        # training loop
        for ep in range(tot_epoch):
            print('EP:%s'%(ep))
            if flag_adam == True:
                flag_optimizer = True
                flag_prior = False
            else:
                flag_optimizer = (ep < (tot_epoch / 2))
                flag_prior = True



            Acc_ELBO = 0
            for idx, data in enumerate(train_loader):
                if iter_counter < tot_epoch * int(data_N / 100) - 100:
                    flag_encoder_update = True
                    flag_record_memory = True
                else:
                    flag_encoder_update = False
                    flag_record_memory = False



                Adam_encoder.zero_grad()
                x, _ = data
                x=x.cuda()
                x=x.view(-1,784)

                # initialize the SGHMC

                if ep == 0:
                    V, M, r, tau, g = self._init_mat()
                    W_dict = self.model.decoder._get_W_dict()
                # ELBO
                ELBO = self._compute_ELBO(x, data_N, W_dict, sigma_out=sigma_out, W_mean_prior=0., W_sigma_prior=1.,
                                          record_memory=flag_record_memory, flag_prior=flag_prior)
                Acc_ELBO += ELBO
                # Backwards
                ELBO.backward()

                G_dict = self.model.decoder._get_grad_W_dict()

                # SGHMC step

                W_dict, r, V, M = self._SGHMC_step(W_dict=W_dict, G_dict=G_dict, eps=eps, eps2=eps, M=M, V=V, r=r,
                                                   flag_optimizer=flag_optimizer, counter=counter,
                                                   flag_compensator=False, update_noise=0, flag_adam=flag_adam)
                counter += 1


                if flag_encoder_update:
                    list_grad_p_z = grap_modify_grad(list_p_z, 1, data_N)
                    zero_grad(list_p_z)
                    assign_grad(list_p_z, list_grad_p_z)
                    Adam_encoder.step()

                # Training progress:

                if iter_counter < thinning or flag_optimizer:
                    W_sample['sample_1'] = copy.deepcopy(W_dict)

                if (iter_counter + 1) % thinning == 0 and not flag_optimizer:
                    sample_counter += 1
                    W_sample = Update_W_sample(W_dict, W_sample, sample_counter, maxsize=max_sample_size)

                if flag_results == True and (iter_counter + 1) % result_interval == 0:
                    print('Training: iter:%s ELBO:%s' % (iter_counter+1, (Acc_ELBO / (784*100*result_interval)).cpu().data.numpy()))
                    Acc_ELBO=0
                    # RMSE, MAE, NLL = Test_UCI(self.model, self.test_log_likelihood, W_sample, overall_test,
                    #                           sigma_out=sigma_out, split=3)
                    #
                    # print('Test: ep:%s RMSE:%s MAE:%s NLL:%s' % (
                    # ep, RMSE.cpu().data.numpy(), MAE.cpu().data.numpy(), NLL.cpu().data.numpy()))
                    # RMSE_mat[result_counter], MAE_mat[result_counter], NLL_mat[
                    #     result_counter] = RMSE.cpu().data.numpy(), MAE.cpu().data.numpy(), NLL.cpu().data.numpy()
                    # result_counter += 1
                iter_counter += 1

    def test_log_likelihood(self,x_test,y_test,W_sample,sigma_out):
        raise NotImplementedError

    def _gaussian_log_likelihood(self,x,decode,sigma_out=0.1):
        # decode is Nz x N x out
        x_expand = torch.unsqueeze(x, dim=0).repeat(decode.shape[0], 1, 1) # Nz x N x out

        log_likelihood = -0.5 * torch.sum((x_expand - decode) ** 2 / (sigma_out ** 2),
                                          dim=2)- 0.5 * self.model.output_dim * (
                                             math.log(sigma_out ** 2) + math.log(2 * np.pi)) # N_z x N

        return log_likelihood

    def _compute_ELBO(self,x,data_N,W_dict,sigma_out=0.1,W_mean_prior=0.,W_sigma_prior=1.,record_memory=False,flag_prior=True):
        if flag_prior==False:
            W_const=0.
        else:
            W_const=1.

        num_X = x.shape[0]
        output_dim = x.shape[1]
        z_recog,mean_recog,sigma_recog=self.model.sample_latent_variable(x)
        if not record_memory:
            z_recog = z_recog.clone().detach()
            mean_recog,sigma_recog=mean_recog.clone().detach(),sigma_recog.clone().detach()

        #assign the W_dict
        self.model.decoder._assign_weight(W_dict)

        decode_x = self.model.decoding(z_recog) # Nz x N x out

        log_likelihood=self._gaussian_log_likelihood(x,decode_x,sigma_out=sigma_out) # Nz x N

        ELBO=log_likelihood-self.model.KL_coef*torch.unsqueeze(self.model._KL_encoder(mean_recog,sigma_recog),dim=0)
        ELBO=torch.sum(torch.mean(ELBO,dim=0))/num_X*data_N+W_const*self._compute_W_prior(W_mean_prior=W_mean_prior,W_sigma_prior=W_sigma_prior)


        return ELBO