# Import the Lib
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
class fc_Encoder_Decoder(nn.Module):
    '''
    This class implements the base network structure for fully connected encoder or decoder.
    This also support the dynamically adding new nodes at the output layer for decoder.
    '''
    def __init__(self,input_dim,output_dim,hidden_layer_num=2,hidden_unit=[100,50],activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False,flag_LV=False,output_const=1.,add_const=0.):
        '''
        Init method
        :param input_dim: The input dimensions
        :type input_dim: int
        :param output_dim: The output dimension of the network
        :type output_dim: int
        :param hidden_layer_num: The number of hidden layers excluding the output layer
        :type hidden_layer_num: int
        :param hidden_unit: The hidden unit size
        :type hidden_unit: list
        :param activations: The activation function for hidden layers
        :type activations: string
        :param flag_only_output_layer: If we only use output layer, so one hidden layer nerual net
        :type flag_only_output_layer: bool
        :param drop_out_rate: The disable percentage of the hidden node
        :param flag_drop_out: Bool, whether to use drop out
        '''
        super(fc_Encoder_Decoder,self).__init__()
        self.drop_out_rate=drop_out_rate
        self.flag_drop_out=flag_drop_out
        self.output_dim=output_dim
        self.input_dim=input_dim
        self.hidden_layer_num=hidden_layer_num
        self.hidden_unit=hidden_unit
        self.flag_only_output_layer=flag_only_output_layer
        self.flag_LV=flag_LV
        self.output_const=output_const
        self.add_const=add_const
        if self.flag_LV==True:
            #raise NotImplementedError('LV is not implemented yet')
            self.LV_dim=output_dim
        self.enable_output_act=False
        self.drop_out=nn.Dropout(self.drop_out_rate)
        # activation functions
        self.activations = activations
        if activations=='ReLU':
            self.act=F.relu
        elif activations=='Sigmoid':
            self.act=F.sigmoid
        elif activations=='Tanh':
            self.act=F.tanh
        elif activations=='Elu':
            self.act=F.elu
        elif activations=='Selu':
            self.act=F.selu
        else:
            raise NotImplementedError

        if activations_output=='ReLU':
            self.enable_output_act=True
            self.act_out=F.relu
        elif activations_output=='Sigmoid':
            self.enable_output_act = True
            self.act_out=F.sigmoid
        elif activations_output=='Tanh':
            self.enable_output_act = True
            self.act_out=F.tanh
        elif activations_output=='Elu':
            self.enable_output_act = True
            self.act_out=F.elu
        elif activations_output=='Selu':
            self.enable_output_act = True
            self.act_out=F.selu
        elif activations_output=='Softplus':
            self.enable_output_act = True
            self.act_out=F.softplus

        # whether to use multi NN or single layer NN
        if self.flag_only_output_layer==False:
            assert len(self.hidden_unit)==hidden_layer_num,'Hidden layer unit length %s inconsistent with layer number %s'%(len(self.hidden_unit),self.hidden_layer_num)

            # build hidden layers
            self.hidden=nn.ModuleList()
            for layer_ind in range(self.hidden_layer_num):
                if layer_ind==0:
                    self.hidden.append(nn.Linear(self.input_dim,self.hidden_unit[layer_ind]))
                else:
                    self.hidden.append(nn.Linear(self.hidden_unit[layer_ind-1],self.hidden_unit[layer_ind]))


            # output layer
            self.out=nn.Linear(self.hidden_unit[-1],self.output_dim)
            if self.flag_LV:
                self.out_LV=nn.Linear(self.hidden_unit[-1],self.LV_dim)


            # Xavier initializer
            # for layers in self.hidden:
            #     torch.nn.init.xavier_uniform(layers.weight)
        else:
            self.out=nn.Linear(self.input_dim,self.output_dim)
            if self.flag_LV:
                self.out_LV = nn.Linear(self.hidden_unit[-1], self.LV_dim)

        #torch.nn.init.xavier_uniform(self.out.weight)
    def _assign_weight(self,W_dict):
        if self.flag_only_output_layer==False:
            for layer_ind in range(self.hidden_layer_num):
                layer_weight=W_dict['weight_layer_%s'%(layer_ind)]
                layer_bias=W_dict['bias_layer_%s'%(layer_ind)]

                self.hidden[layer_ind].weight=torch.nn.Parameter(layer_weight.data)
                self.hidden[layer_ind].bias=torch.nn.Parameter(layer_bias.data)
            out_weight=W_dict['weight_out']
            out_bias=W_dict['bias_out']
            self.out.weight=torch.nn.Parameter(out_weight.data)
            self.out.bias=torch.nn.Parameter(out_bias.data)
            if self.flag_LV:
                out_weight_LV = W_dict['weight_out_LV']
                out_bias_LV = W_dict['bias_out_LV']
                self.out_LV.weight = torch.nn.Parameter(out_weight_LV.data)
                self.out_LV.bias = torch.nn.Parameter(out_bias_LV.data)

        else:
            out_weight = W_dict['weight_out']
            out_bias = W_dict['bias_out']
            self.out.weight = torch.nn.Parameter(out_weight.data)
            self.out.bias = torch.nn.Parameter(out_bias.data)
            if self.flag_LV:
                out_weight_LV = W_dict['weight_out_LV']
                out_bias_LV = W_dict['bias_out_LV']
                self.out_LV.weight = torch.nn.Parameter(out_weight_LV.data)
                self.out_LV.bias = torch.nn.Parameter(out_bias_LV.data)

    def _get_W_dict(self):
        W_dict=collections.OrderedDict()
        if self.flag_only_output_layer == False:
            for layer_ind in range(self.hidden_layer_num):
                W_dict['weight_layer_%s'%(layer_ind)]=self.hidden[layer_ind].weight.clone().detach()
                W_dict['bias_layer_%s'%(layer_ind)]=self.hidden[layer_ind].bias.clone().detach()
            W_dict['weight_out']=self.out.weight.clone().detach()
            W_dict['bias_out']=self.out.bias.clone().detach()
            if self.flag_LV:
                W_dict['weight_out_LV'] = self.out_LV.weight.clone().detach()
                W_dict['bias_out_LV'] = self.out_LV.bias.clone().detach()
        else:
            W_dict['weight_out'] = self.out.weight.clone().detach()
            W_dict['bias_out'] = self.out.bias.clone().detach()
            if self.flag_LV:
                W_dict['weight_out_LV'] = self.out_LV.weight.clone().detach()
                W_dict['bias_out_LV'] = self.out_LV.bias.clone().detach()
        return W_dict
    def _get_grad_W_dict(self):
        G_dict=collections.OrderedDict()
        if self.flag_only_output_layer == False:
            for layer_ind in range(self.hidden_layer_num):
                if self.hidden[layer_ind].weight.grad is not None:
                    G_dict['weight_layer_%s'%(layer_ind)]=-self.hidden[layer_ind].weight.grad.clone().detach()
                    G_dict['bias_layer_%s'%(layer_ind)]=-self.hidden[layer_ind].bias.grad.clone().detach()
            G_dict['weight_out']=-self.out.weight.grad.clone().detach()
            G_dict['bias_out']=-self.out.bias.grad.clone().detach()
            if self.flag_LV:
                G_dict['weight_out_LV'] = -self.out_LV.weight.grad.clone().detach()
                G_dict['bias_out_LV'] = -self.out_LV.bias.grad.clone().detach()
        else:
            G_dict['weight_out'] = -self.out.weight.grad.clone().detach()
            G_dict['bias_out'] = -self.out.bias.grad.clone().detach()
            if self.flag_LV:
                G_dict['weight_out_LV'] = -self.out_LV.weight.grad.clone().detach()
                G_dict['bias_out_LV'] = -self.out_LV.bias.grad.clone().detach()
        return G_dict
    def _flatten_stat(self):
        if self.flag_only_output_layer==False:
            for idx,layer in enumerate(self.hidden):
                W_weight,b_weight=layer.weight.view(1,-1),layer.bias.view(1,-1) # 1 x dim
                weight_comp=torch.cat((W_weight,b_weight),dim=1) # 1 x dim
                if idx==0:
                    weight_flat=weight_comp
                else:
                    weight_flat=torch.cat((weight_flat,weight_comp),dim=1)

            # Output layer (need to account for the mask)
            Out_weight,Out_b_weight=self.out.weight,self.out.bias # N_in x N_out or N_out

            if self.flag_LV==True:
                W_weight_LV,b_weight_LV=self.out_LV.weight,self.out_LV.bias
                Out_weight,Out_b_weight=torch.cat((Out_weight,W_weight_LV),dim=1),torch.cat((Out_b_weight,b_weight_LV),dim=0) #' This should be N_in x (2 x N_out) or (2 X N_out)'

            return weight_flat,Out_weight,Out_b_weight
        else:
            Out_weight, Out_b_weight = self.out.weight, self.out.bias  # N_in x N_out or N_out
            if self.flag_LV == True:
                W_weight_LV, b_weight_LV = self.out_LV.weight, self.out_LV.bias
                Out_weight, Out_b_weight = torch.cat((Out_weight, W_weight_LV), dim=1), torch.cat(
                    (Out_b_weight, b_weight_LV), dim=0)  # ' This should be N_in x (2 x N_out) or (2 X N_out)'

            return [],Out_weight, Out_b_weight


    def forward(self,x):
        '''
        The forward pass
        :param x: Input Tensor
        :type x: Tensor
        :return: output from the network
        :rtype: Tensor
        '''
        min_sigma=-4.6
        max_sigma=2
        if self.flag_only_output_layer==False:
            for layer in self.hidden:

                x=self.act(layer(x))
                if self.flag_drop_out:
                    x=self.drop_out(x)
            if self.enable_output_act==True:
                output=self.act_out(self.out(x))
                if self.flag_LV:
                    output_LV=self.act_out(self.out_LV(x))
                    # clamp
                    output_LV=torch.clamp(output_LV,min=min_sigma,max=max_sigma) # Corresponds to 0.1 sigma
            else:
                output=self.out(x)
                if self.flag_LV:
                    output_LV=self.out_LV(x)
                    # clamp
                    output_LV = torch.clamp(output_LV, min=min_sigma,max=max_sigma)  # Corresponds to 0.1 sigma


        else:
            if self.enable_output_act==True:
                output=self.act_out(self.out(x))
                if self.flag_LV:
                    output_LV=self.act_out(self.out_LV(x))
                    # clamp
                    output_LV = torch.clamp(output_LV, min=min_sigma,max=max_sigma)  # Corresponds to 0.1 sigma
            else:
                output=self.out(x)
                if self.flag_LV:
                    output_LV=self.out_LV(x)
                    # clamp
                    output_LV = torch.clamp(output_LV, min=min_sigma,max=max_sigma)  # Corresponds to 0.1 sigma

        output=self.add_const+self.output_const*output
        if self.flag_LV:
            output=torch.cat((output,output_LV),dim=-1)
        return output

    def _add_new_node_last_random(self,new_dimension):
        '''
        This is to add new node at the output layer (Used for streaming train only)
        :param new_dimension:
        :type new_dimension:
        :return:
        :rtype:
        '''
        if self.flag_LV:
            raise NotImplementedError
        current_out_weight=self.out.weight.data
        current_out_bias=self.out.bias.data

        # new weight matrix and bias
        w_new=0.01*torch.randn(new_dimension,self.hidden_unit[-1])
        bias_new=0.01*torch.randn(new_dimension)
        # Initialize new weight and bias

        # nn.init.xavier_uniform_(w_new)
        # bound = 1./np.sqrt(nn.init._calculate_fan_in_and_fan_out(w_new)[0])
        # nn.init.uniform_(bias_new,-bound,bound)

        # concat to old matrix and bias
        w_new_mat=torch.cat((current_out_weight,w_new),dim=0)

        bias_new_mat=torch.cat((current_out_bias,bias_new),dim=0)

        # set current weight/bias to new weight/bias
        if self.flag_only_output_layer == False:
            self.out=nn.Linear(self.hidden_unit[-1],self.output_dim+new_dimension)
        else:
            self.out = nn.Linear(self.input_dim, self.output_dim+new_dimension)

        self.out.weight.data=torch.tensor(w_new_mat.data,requires_grad=True)
        self.out.weight.bias=torch.tensor(bias_new_mat.data,requires_grad=True)
