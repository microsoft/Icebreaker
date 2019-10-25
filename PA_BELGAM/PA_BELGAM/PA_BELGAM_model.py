from Util.Util_func import *
from base_Model.base_Network import *
import numpy as np
import copy
#
class PA_BELGAM(object):
    def __init__(self,latent_dim,input_dim,output_dim,prior_net_layer_num=2,prior_net_hidden=[20,20],recog_net_layer_num=2,recog_net_hidden=[20,20],decoder_layer_num=2,decoder_hidden=[20,20],
                 output_const=1.,add_const=0.,sample_Z=50,KL_coef=1.,hybrid_coef=0.5,flag_log_q=True):
        # store parameters
        self.latent_dim=latent_dim
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.prior_net_layer_num,self.prior_net_hidden=prior_net_layer_num,prior_net_hidden
        self.recog_net_layer_num,self.recog_net_hidden=recog_net_layer_num,recog_net_hidden
        self.decoder_layer_num,self.decoder_hidden=decoder_layer_num,decoder_hidden
        self.output_const=output_const
        self.add_const=add_const
        self.sample_Z=sample_Z
        self.KL_coef=KL_coef
        self.hybrid_coef=hybrid_coef
        self.flag_log_q=flag_log_q
        # Init model component
        # Prior net
        self.prior_net=fc_Encoder_Decoder(self.input_dim,2*self.latent_dim,hidden_layer_num=self.prior_net_layer_num,hidden_unit=self.prior_net_hidden,
                                          activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False
                                        )

        # Recog net
        self.recog_net=fc_Encoder_Decoder(self.input_dim+self.output_dim,2*self.latent_dim,hidden_layer_num=self.recog_net_layer_num,hidden_unit=self.recog_net_hidden,
                                          activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False
                                        )

        # Decoder p(y|x,z)
        self.decoder=fc_Encoder_Decoder(self.latent_dim+self.input_dim,self.output_dim,hidden_layer_num=self.decoder_layer_num,hidden_unit=self.decoder_hidden,
                                          activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False
                                        )

    def _extract_state_dict(self):
        decoder_state=copy.deepcopy(self.decoder.state_dict())
        prior_net_state=copy.deepcopy(self.prior_net.state_dict())
        recog_net_state=copy.deepcopy(self.recog_net)
        return prior_net_state,recog_net_state,decoder_state
    def _load_state_dict(self,prior_net_state,recog_net_state,decoder_state):
        self.prior_net.load_state_dict(prior_net_state)
        self.recog_net.load_state_dict(recog_net_state)
        self.decoder.load_state_dict(decoder_state)
    def _encoding(self,x,y=None,flag_prior_net=True):
        # prior net q(z|x), recog net q(z|x,y)
        # x is N x input_dim y is N x out_dim
        if flag_prior_net==False:
            if y is None:
                raise TypeError('If flag_prior_net is True, y cannot be None')
        if flag_prior_net==True:
            output=self.prior_net.forward(x) # N x 2*latent
        else:
            x_aug=torch.cat((x,y),dim=-1) # N x input+output
            output=self.recog_net.forward(x_aug)
        return output
    def sample_latent_variable(self,x,y=None,flag_prior_net=True):
        batch_size=x.shape[0]
        encoding=self._encoding(x,y,flag_prior_net)
        mean = encoding[:, :self.latent_dim]
        if self.flag_log_q==True:
            sigma = torch.clamp(torch.sqrt(torch.exp(encoding[:, self.latent_dim:])), min=1e-5, max=300.)
        else:
            sigma = torch.clamp(torch.sqrt(torch.clamp((encoding[:, self.latent_dim:]) ** 2,min=1e-8)),min=1e-5,max=300.)

        eps = torch.randn(self.sample_Z, batch_size, self.latent_dim, requires_grad=True)  # size x N  x latent_dim
        z = torch.unsqueeze(mean, dim=0) + torch.unsqueeze(sigma, dim=0) * eps  # size x N x latent_dim

        return z, mean,sigma
    def _KL_encoder(self,mean_prior,sigma_prior,mean_recog,sigma_recog):

        # They are all N x latent
        KL_z = 0.5*(2*torch.log(sigma_prior)-2*torch.log(sigma_recog)-1+(sigma_recog/sigma_prior)**2+(mean_prior-mean_recog)**2/(sigma_prior**2)) # N x latent
        KL_z=torch.sum(KL_z,dim=-1) # N
        return KL_z  # N
    def decoding(self,x,z):
        # x is N x input z is size x N x latent
        x_expand=torch.unsqueeze(x,dim=0).repeat(self.sample_Z,1,1) # size x N x input
        z_aug=torch.cat((x_expand,z),dim=-1) # size x N x latent+input
        output=self.decoder.forward(z_aug) # size x N x output

        return output
    def completion(self,x,W_sample):
        x=x.clone().detach()
        with torch.no_grad():
            z,_,_=self.sample_latent_variable(x,flag_prior_net=True)

        counter = 0
        for key_W, value_W in W_sample.items():
            self.decoder._assign_weight(value_W)
            with torch.no_grad():
                decode=self.decoding(x,z)
            if counter == 0:
                decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                output = decode
            else:
                decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                output = torch.cat((output, decode), dim=0)  # N_W x N_z x N x obs_dim
            counter += 1

        return output



class PA_BELGAM_MNIST(object):
    def __init__(self,latent_dim,input_dim,output_dim,recog_net_layer_num=1,recog_net_hidden=[400],decoder_layer_num=1,decoder_hidden=[400],
                 output_const=1.,add_const=0.,sample_Z=50,KL_coef=1.,flag_log_q=True):
        # store parameters
        self.latent_dim=latent_dim
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.recog_net_layer_num,self.recog_net_hidden=recog_net_layer_num,recog_net_hidden
        self.decoder_layer_num,self.decoder_hidden=decoder_layer_num,decoder_hidden
        self.output_const=output_const
        self.add_const=add_const
        self.sample_Z=sample_Z
        self.KL_coef=KL_coef
        self.flag_log_q=flag_log_q
        # Init model component
        # Recog net q(z|x)
        self.recog_net=fc_Encoder_Decoder(self.input_dim,2*self.latent_dim,hidden_layer_num=self.recog_net_layer_num,hidden_unit=self.recog_net_hidden,
                                          activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False
                                        )

        # Decoder p(y|z)
        self.decoder=fc_Encoder_Decoder(self.latent_dim,self.output_dim,hidden_layer_num=self.decoder_layer_num,hidden_unit=self.decoder_hidden,
                                          activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False
                                        )

    def _extract_state_dict(self):
        decoder_state=copy.deepcopy(self.decoder.state_dict())
        recog_net_state=copy.deepcopy(self.recog_net)
        return recog_net_state,decoder_state
    def _load_state_dict(self,prior_net_state,recog_net_state,decoder_state):
        self.recog_net.load_state_dict(recog_net_state)
        self.decoder.load_state_dict(decoder_state)

    def _encoding(self,x):
        output=self.recog_net.forward(x)
        return output

    def sample_latent_variable(self,x):
        batch_size=x.shape[0]
        encoding=self._encoding(x)
        mean = encoding[:, :self.latent_dim]
        if self.flag_log_q==True:
            sigma = torch.clamp(torch.sqrt(torch.exp(encoding[:, self.latent_dim:])), min=1e-5, max=300.)
        else:
            sigma = torch.clamp(torch.sqrt(torch.clamp((encoding[:, self.latent_dim:]) ** 2,min=1e-8)),min=1e-5,max=300.)

        eps = torch.randn(self.sample_Z, batch_size, self.latent_dim, requires_grad=True)  # size x N  x latent_dim
        z = torch.unsqueeze(mean, dim=0) + torch.unsqueeze(sigma, dim=0) * eps  # size x N x latent_dim

        return z, mean,sigma

    def _KL_encoder(self,mean_recog,sigma_recog):

        # They are all N x latent
        KL_z = 0.5*(2*0-2*torch.log(sigma_recog)-1+(sigma_recog/1.)**2+(0-mean_recog)**2/(1**2)) # N x latent
        KL_z=torch.sum(KL_z,dim=-1) # N
        return KL_z  # N

    def decoding(self,z):
        output=self.decoder.forward(z) # size x N x output

        return output
    def prior_completion(self,size,W_sample):
        eps = torch.randn(size, self.latent_dim)
        counter = 0
        for key_W, value_W in W_sample.items():
            self.decoder._assign_weight(value_W)
            with torch.no_grad():
                decode = self.decoding(eps)
            if counter == 0:
                decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                output = decode
            else:
                decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                output = torch.cat((output, decode), dim=0)  # N_W x N_z x N x obs_dim
            counter += 1
        return output
    def completion(self,x,W_sample):
        x=x.clone().detach()
        with torch.no_grad():
            z,_,_=self.sample_latent_variable(x)

        counter = 0
        for key_W, value_W in W_sample.items():
            self.decoder._assign_weight(value_W)
            with torch.no_grad():
                decode=self.decoding(z)
            if counter == 0:
                decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                output = decode
            else:
                decode = torch.unsqueeze(decode, dim=0)  # 1 x N_z x N x obs_dim
                output = torch.cat((output, decode), dim=0)  # N_W x N_z x N x obs_dim
            counter += 1

        return output

