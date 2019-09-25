import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from base_Model.base_Network import *


class MyDict(collections.OrderedDict):
    def __missing__(self, key):
        val = self[key] = MyDict()
        return val


class BNN_Layer(nn.Module):
    '''
    This implements the Bayesian MLP layer, used in mean-field VI
    '''
    def __init__(self,n_input,n_output,sigma_prior,act_func='ReLU',init_range=0.01,coef_sample=1.):
        super(BNN_Layer,self).__init__()
        self.n_input=n_input
        self.n_output=n_output
        self.sigma_prior=sigma_prior # This is the prior of the weight and bias (scalar)
        self.act_func=act_func
        self.flag_none=False
        self.coef_sample=coef_sample
        if self.act_func=='ReLU':
            self.act=F.relu
        elif self.act_func=='Sigmoid':
            self.act=F.sigmoid
        elif self.act_func=='Tanh':
            self.act=F.tanh
        elif self.act_func=='Softplus':
            self.act=F.softplus
        elif type(self.act_func)==type(None):
            self.flag_none=True
        else:
            raise NotImplementedError
        self.lpw=0
        self.lqw=0
        # Initialize the weight and bias mean/var
        #self.W_mu=nn.Parameter(torch.nn.init.xavier_uniform(torch.Tensor(n_input,n_output).normal_(0,init_range)))
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, init_range))
        self.b_mu=nn.Parameter(torch.Tensor(n_output).uniform_(-init_range,init_range))

        # self.W_sigma=nn.Parameter(F.softplus(torch.Tensor(n_input,n_output).normal_(0,init_range)))
        # self.b_sigma=nn.Parameter(F.softplus(torch.Tensor(n_output).uniform_(-init_range,init_range)))

        self.W_sigma = nn.Parameter(torch.Tensor(n_input, n_output).uniform_(0, 0.3))
        self.b_sigma = nn.Parameter(torch.Tensor(n_output).uniform_(0, 0.3))

    def forward(self,X,flag_mean=False):
        if flag_mean:
            output=torch.matmul(X,self.W_mu)+self.b_mu.expand(X.size()[0],self.n_output)
            if self.flag_none==False:
                output=self.act(output)
            return output
        eps_w,eps_b=self._generate_eps()
        W=self.W_mu+self.coef_sample*self.W_sigma*eps_w
        b=self.b_mu+self.coef_sample*self.b_sigma*eps_b

        #output=torch.matmul(X,W)+b.expand(X.size()[0],self.n_output)
        output=F.linear(X,W.t(),b)
        if self.flag_none==False:
            output=self.act(output)
        #self.lpw=self._log_gaussian(W,0,self.sigma_prior).sum()+self._log_gaussian(b,0,self.sigma_prior).sum()
        #self.lqw=self._log_gaussian_logsigma(W,self.W_mu,self.W_logsigma).sum()+self._log_gaussian_logsigma(b,self.b_mu,self.b_logsigma).sum()
        return output,W,b
    def draw_weights(self):
        eps_w, eps_b = self._generate_eps()
        W = self.W_mu + self.coef_sample*self.W_sigma * eps_w
        b = self.b_mu + self.coef_sample*self.b_sigma * eps_b
        return W,b
    def _forward_weight(self,X,W,b):
        output = torch.matmul(X, W) + b.expand(X.size()[0], self.n_output)
        if self.flag_none==False:
            output=self.act(output)
        #lpw = self._log_gaussian(W, 0, self.sigma_prior).sum() + self._log_gaussian(b, 0, self.sigma_prior).sum()
        #lqw = self._log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + self._log_gaussian_logsigma(b,
                                                                                                                  #self.b_mu,
                                                                                                                  #self.b_logsigma).sum()
        return output

    def _generate_eps(self):
        #raise NotImplementedError
        return torch.Tensor(self.n_input,self.n_output).normal_(0,1.),torch.Tensor(self.n_output).normal_(0,1.)
    def _log_gaussian(self,x,mu,sigma):
        raise NotImplementedError
        return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2) # N x D
    def _log_gaussian_logsigma(self,x,mu,logsigma):
        raise NotImplementedError
        return float(-0.5 * np.log(2 * np.pi)) - torch.log(F.softplus(logsigma)) - (x - mu)**2 / (2 * F.softplus(logsigma)**2) # N x D
class BNN_local_Layer(nn.Module):
    # Implementation of BNN local re-parametrization layer
    def __init__(self,n_input,n_output,sigma_prior,act_func='ReLU',init_range=0.01,coef_sample=1.):
        super(BNN_local_Layer,self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior  # This is the prior of the weight and bias (scalar)
        self.act_func = act_func
        self.flag_none = False
        self.coef_sample = coef_sample
        if self.act_func == 'ReLU':
            self.act = F.relu
        elif self.act_func == 'Sigmoid':
            self.act = F.sigmoid
        elif self.act_func == 'Tanh':
            self.act = F.tanh
        elif self.act_func == 'Softplus':
            self.act = F.softplus
        elif type(self.act_func) == type(None):
            self.flag_none = True
        else:
            raise NotImplementedError
        self.lpw = 0
        self.lqw = 0
        # Initialize the weight and bias mean/var

        # self.W_mu=nn.Parameter(torch.nn.init.xavier_uniform(torch.Tensor(n_input,n_output).normal_(0,init_range)))
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, init_range))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, init_range))

        # self.W_sigma=nn.Parameter(F.softplus(torch.Tensor(n_input,n_output).normal_(0,init_range)))
        # self.b_sigma=nn.Parameter(F.softplus(torch.Tensor(n_output).uniform_(-init_range,init_range)))

        self.W_sigma = nn.Parameter(torch.Tensor(n_input, n_output).uniform_(0, 1))
        self.b_sigma = nn.Parameter(torch.Tensor(n_output).uniform_(0, 1))
    def _generate_eps(self,shape):
        #raise NotImplementedError
        return torch.Tensor(shape).normal_(0,1.)
    def forward(self,X):

        mean=F.linear(X,self.W_mu.t(),self.b_mu)
        sigma_square=F.linear(X**2,(self.W_sigma.t())**2,self.b_sigma**2)
        eps = self._generate_eps(mean.shape)  # N x output_dim
        output=mean+self.coef_sample*torch.sqrt(sigma_square)*eps
        if self.flag_none==False:
            output=self.act(output)
        return output,self.W_mu,self.b_mu

class BNN_mean_field(nn.Module):
    # BNN Mean field MLP
    def __init__(self,input_dim,output_dim,hidden_layer_num=2,hidden_unit=[100,50],activations='ReLU',activations_output=None,flag_only_output_layer=False,sigma_prior=0.1, init_range=0.01,coef_sample=1.,flag_LV=False,LV_act_output=None):
        super(BNN_mean_field,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_layer_num=hidden_layer_num
        self.hidden_unit=hidden_unit
        self.act=activations
        self.act_out=activations_output
        self.flag_only_output_layer=flag_only_output_layer
        self.sigma_prior=sigma_prior
        self.init_range=init_range
        self.coef_sample=coef_sample
        # Learned Variance
        self.flag_LV=flag_LV
        self.LV_act_out=LV_act_output
        if self.flag_only_output_layer==False:
            assert len(self.hidden_unit)==self.hidden_layer_num,'The hidden layer number is not consistent with hidden units'
            self.hidden=nn.ModuleList()
            for idx in range(self.hidden_layer_num):
                if idx ==0:
                    n_input=input_dim
                    n_output=hidden_unit[0]
                else:
                    n_input = hidden_unit[idx-1]
                    n_output = hidden_unit[idx]
                self.hidden.append(BNN_Layer(n_input,n_output,self.sigma_prior,act_func=self.act,init_range=init_range,coef_sample=coef_sample))



            self.out=BNN_Layer(hidden_unit[-1],output_dim,self.sigma_prior,act_func=self.act_out,init_range=init_range,coef_sample=coef_sample)
            if self.flag_LV==True:
                self.out2=BNN_Layer(hidden_unit[-1],output_dim,self.sigma_prior,act_func=self.LV_act_out,init_range=init_range,coef_sample=coef_sample)
        else:
            self.out=BNN_Layer(input_dim,output_dim,self.sigma_prior,act_func=self.act_out,init_range=init_range,coef_sample=coef_sample)
            if self.flag_LV==True:
                self.out2=BNN_Layer(input_dim,output_dim,self.sigma_prior,act_func=self.LV_act_out,init_range=init_range,coef_sample=coef_sample)

        self.weight_dict = MyDict()# This is used for storing the weight samples with structure times, layers (from low to high), w, b
    def _flatten_weight(self):
        '''
        This is to form a 2D tensor for weight samples from the BNN. The structure is sample_number x (weight_layer_1+bias_layer_1+weight_layer_2+bias_layer_2+...) '+' means concatenation
        :return: 2D tensor
        :rtype: Tensor
        '''
        # Check if weight_dict is empty
        assert self.weight_dict,'Weight_dict is empty'
        # Iterate through the dict (sample number)
        counter_sample=0
        for key,value in self.weight_dict.items():
            # Iterate through (layer)
            counter_layer=0
            for key_layer,value_layer in value.items():
                if counter_layer==0:
                    W_comp=value_layer['W'].view(1,-1) # 1 x dim_W
                    b_comp=value_layer['b'].view(1,-1) # 1 x dim_b
                    layer_comp=torch.cat((W_comp,b_comp),dim=1) # 1 x (dim_W+dim_b)
                    layer=layer_comp
                    counter_layer+=1
                else:
                    W_comp = value_layer['W'].view(1, -1)  # 1 x dim_W
                    b_comp = value_layer['b'].view(1, -1)  # 1 x dim_b
                    layer_comp = torch.cat((W_comp, b_comp), dim=1)  # 1 x (dim_W+dim_b)
                    layer=torch.cat((layer,layer_comp),dim=1)
                    counter_layer+=1
            # Cumulate samples
            if counter_sample==0:
                flat_weight=layer
                counter_sample+=1
            else:
                flat_weight=torch.cat((flat_weight,layer),dim=0)
                counter_sample+=1

        # Clear the weight dict for momery
        self.weight_dict=MyDict()
        return flat_weight
    def _flatten_stat(self):
        if self.flag_only_output_layer==False:
            for idx,layer in enumerate(self.hidden):
                W_mean,b_mean,W_sigma,b_sigma=layer.W_mu.view(1,-1),layer.b_mu.view(1,-1),layer.W_sigma.view(1,-1),layer.b_sigma.view(1,-1) # 1 x dim
                mean_comp=torch.cat((W_mean,b_mean),dim=1) # 1 x dim
                sigma_comp=torch.cat((W_sigma,b_sigma),dim=1) # 1 x dim
                if idx==0:
                    mean_flat=mean_comp
                    sigma_flat=sigma_comp
                else:
                    mean_flat=torch.cat((mean_flat,mean_comp),dim=1)
                    sigma_flat=torch.cat((sigma_flat,sigma_comp),dim=1) # 1 x dim
            # Output layer (need to account for the mask)
            W_mean,b_mean,W_sigma,b_sigma=self.out.W_mu,self.out.b_mu,self.out.W_sigma,self.out.b_sigma # N_in x N_out or N_out
            if self.flag_LV==True:
                W_mean_LV,b_mean_LV,W_sigma_LV,b_sigma_LV=self.out2.W_mu,self.out2.b_mu,self.out2.W_sigma,self.out2.b_sigma
                W_mean,W_sigma,b_mean,b_sigma=torch.cat((W_mean,W_mean_LV),dim=1),torch.cat((W_sigma,W_sigma_LV),dim=1),torch.cat((b_mean,b_mean_LV),dim=0),torch.cat((b_sigma,b_sigma_LV),dim=0) #' This should be N_in x (2 x N_out) or (2 X N_out)'

            return mean_flat,sigma_flat,W_mean,b_mean,W_sigma,b_sigma


        else:
            # Output layer (need to account for the mask)
            W_mean, b_mean, W_sigma, b_sigma = self.out.W_mu, self.out.b_mu, self.out.W_sigma, self.out.b_sigma  # N_in x N_out or N_out
            if self.flag_LV==True:
                W_mean_LV,b_mean_LV,W_sigma_LV,b_sigma_LV=self.out2.W_mu,self.out2.b_mu,self.out2.W_sigma,self.out2.b_sigma
                W_mean,W_sigma,b_mean,b_sigma=torch.cat((W_mean,W_mean_LV),dim=1),torch.cat((W_sigma,W_sigma_LV),dim=1),torch.cat((b_mean,b_mean_LV),dim=0),torch.cat((b_sigma,b_sigma_LV),dim=0) #' This should be N_in x (2 x N_out) or (2 X N_out)'

            return W_mean, b_mean, W_sigma, b_sigma
    def _flatten_out(self):
        raise NotImplementedError
        W_mu=self.out.W_mu # dim x out
        b_mu=self.out.b_mu # out
        W_sigma=self.out.W_sigma # dim x out
        b_sigma=self.out.b_sigma # out
        W_b_mu=torch.cat((W_mu,torch.unsqueeze(b_mu,dim=0)),dim=0) # (dim+1) x out
        W_b_sigma=torch.cat((W_sigma,torch.unsqueeze(b_sigma,dim=0)),dim=0) # (dim+1) x out

        decoder_embedding=torch.cat((W_b_mu,W_b_sigma),dim=0) # (dim+1)x2 x out
        #decoder_embedding=W_b_sigma
        return torch.unsqueeze(decoder_embedding.t(),dim=0) # 1 x out x (dim+1)x2



    def _forward(self,X,sample_number,flag_record=False):
        '''
        Forward Pass
        :param X:
        :type X:
        :param flag_record:
        :type flag_record:
        :return:
        :rtype:
        '''
        min_sigma=-4.6
        if self.flag_only_output_layer==False:
            for idx,layers in enumerate(self.hidden):
                X,W,b=layers.forward(X)
                if flag_record==True:
                    self.weight_dict['sample_%s'%(sample_number)]['layer_%s'%(idx)]['W']=W
                    self.weight_dict['sample_%s' % (sample_number)]['layer_%s' % (idx)]['b'] = b
            output,W,b=self.out(X)
            if self.flag_LV==True:
                output_LV,W_LV,b_LV=self.out2(X)
                #clamp
                output_LV=torch.clamp(output_LV,min=min_sigma)

                output=torch.cat((output,output_LV),dim=-1) # ..... x N x (2 x out_dim)
                W=torch.cat((W,W_LV),dim=-1) # ... x (2 x N_out)
                b=torch.cat((b,b_LV),dim=-1) # ( 2 x N_out)
            if flag_record==True:
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['W'] = W
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['b'] = b
        else:
            output, W, b = self.out(X)
            if self.flag_LV==True:
                output_LV,W_LV,b_LV=self.out2(X)
                #clamp
                output_LV=torch.clamp(output_LV,min=min_sigma)

                output=torch.cat((output,output_LV),dim=-1) # ..... x N x (2 x out_dim)
                W=torch.cat((W,W_LV),dim=-1) # ... x (2 x N_out)
                b=torch.cat((b,b_LV),dim=-1) # ( 2 x N_out)
            if flag_record == True:
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['W'] = W
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['b'] = b
        return output
    def forward(self,X,num_sample=1,flag_record=False):
        for num in range(num_sample):
            output_comp=self._forward(X,num,flag_record=flag_record)
            if num==0:
                output=torch.unsqueeze(output_comp,dim=0) # 1 x N x output_dim
            else:
                output_comp = torch.unsqueeze(output_comp, dim=0)  # 1 x N x output_dim
                output=torch.cat((output,output_comp),dim=0) # num x N x output_dim
        return output
class BNN_last_local(BNN_mean_field):
    # This is NN with only last layer as BNN and rest are deterministic. In this experiment only used for PNP with zero variance on last layer (equivalent to deterministic layer)
    def __init__(self,*args,**kwargs):
        super(BNN_last_local,self).__init__(*args,**kwargs)
        self.act_func=kwargs['activations']
        if self.act_func == 'ReLU':
            self.act = F.relu
        elif self.act_func == 'Sigmoid':
            self.act = F.sigmoid
        elif self.act_func == 'Tanh':
            self.act = F.tanh
        elif self.act_func == 'Softplus':
            self.act = F.softplus
        elif type(self.act_func) == type(None):
            self.flag_none = True
        else:
            raise NotImplementedError
        self.flag_output_layer=self.flag_only_output_layer
        self.flag_only_output_layer=True
        #### NOTE: In this class, we use flag_output_layer for indicator of using only output layer and the original flag_only_output_layer is always set to True for computing the KL divergence (KL_W is only considered at last layer)
        if self.flag_output_layer==False:
            # Det NET + BNN
            assert len(self.hidden_unit)==self.hidden_layer_num, 'The hidden lyaer number is not consistent with hidden units'
            self.hidden=nn.ModuleList()
            for idx in range(self.hidden_layer_num):
                if idx ==0:
                    n_input=self.input_dim
                    n_output=self.hidden_unit[0]
                else:
                    n_input=self.hidden_unit[idx-1]
                    n_output=self.hidden_unit[idx]
                # Note that this only store the linear transformation, so need manually add activation function in forward()
                self.hidden.append(nn.Linear(n_input,n_output))
            # Last layer as BNN
            self.out=BNN_local_Layer(self.hidden_unit[-1],self.output_dim,self.sigma_prior,act_func=self.act_out,init_range=self.init_range,coef_sample=self.coef_sample)
            if self.flag_LV==True:
                self.out2=BNN_local_Layer(self.hidden_unit[-1],self.output_dim,self.sigma_prior,act_func=self.LV_act_out,init_range=self.init_range,coef_sample=self.coef_sample)
        else:
            self.out=BNN_local_Layer(self.input_dim,self.output_dim,self.sigma_prior,act_func=self.act_out,init_range=self.init_range,coef_sample=self.coef_sample)
            if self.flag_LV==True:
                self.out2=BNN_local_Layer(self.input_dim,self.output_dim,self.sigma_prior,act_func=self.LV_act_out,init_range=self.init_range,coef_sample=self.coef_sample)
        self.weight_dict=MyDict() # Store Weight samples

    def _forward(self, X, sample_number, flag_record=False):
        min_sigma=-4.6
        if self.flag_output_layer==False:
            for idx,layers in enumerate(self.hidden):
                X=layers.forward(X)
                # Activation function
                X=self.act(X)
            # Last layer
            output,W,b=self.out(X)
            if self.flag_LV==True:
                output_LV,W_LV,b_LV=self.out2(X)
                # clamp
                output_LV = torch.clamp(output_LV, min=min_sigma)

                output=torch.cat((output,output_LV),dim=-1) # ..... x N x (2 x out_dim)

                W=torch.cat((W,W_LV),dim=-1) # ... x (2 x N_out)
                b=torch.cat((b,b_LV),dim=-1) # ( 2 x N_out)

            if flag_record==True:
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['W'] = W
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['b'] = b
        else:
            output, W, b = self.out(X)
            if self.flag_LV==True:
                output_LV,W_LV,b_LV=self.out2(X)
                # clamp
                output_LV = torch.clamp(output_LV, min=min_sigma)
                output=torch.cat((output,output_LV),dim=-1) # ..... x N x (2 x out_dim)
                W=torch.cat((W,W_LV),dim=-1) # ... x (2 x N_out)
                b=torch.cat((b,b_LV),dim=-1) # ( 2 x N_out)
            if flag_record == True:
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['W'] = W
                self.weight_dict['sample_%s' % (sample_number)]['layer_output']['b'] = b
        return output
    def forward(self,X,num_sample=1,flag_record=False):
        for num in range(num_sample):
            output_comp=self._forward(X,num,flag_record=flag_record)
            if num==0:
                output=torch.unsqueeze(output_comp,dim=0)
            else:
                output_comp=torch.unsqueeze(output_comp,dim=0)

                output = torch.cat((output, output_comp), dim=0)  # num x N x output_dim

        return output

    def _flatten_stat(self):

        # Output layer (need to account for the mask)
        W_mean, b_mean, W_sigma, b_sigma = self.out.W_mu, self.out.b_mu, self.out.W_sigma, self.out.b_sigma  # N_in x N_out or N_out
        if self.flag_LV == True:
            W_mean_LV, b_mean_LV, W_sigma_LV, b_sigma_LV = self.out2.W_mu, self.out2.b_mu, self.out2.W_sigma, self.out2.b_sigma
            W_mean, W_sigma, b_mean, b_sigma = torch.cat((W_mean, W_mean_LV), dim=1), torch.cat((W_sigma, W_sigma_LV),
                                                                                                dim=1), torch.cat(
                (b_mean, b_mean_LV), dim=0), torch.cat((b_sigma, b_sigma_LV),
                                                       dim=0)  # ' This should be N_in x (2 x N_out) or (2 X N_out)'

        return W_mean, b_mean, W_sigma, b_sigma