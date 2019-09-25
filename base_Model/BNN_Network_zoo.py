from base_Model.base_BNN import *
from Util.Util_func import *
from base_Model.base_Network import *
import numpy as np

class Point_Net_Plus_BNN(object):
    def __init__(self,latent_dim,obs_dim,dim_before_agg,encoder_layer_num_before_agg=1,encoder_hidden_before_agg=None,encoder_layer_num_after_agg=1,
               encoder_hidden_after_agg=[100],embedding_dim=10,decoder_layer_num=2,decoder_hidden=[50,100],pooling='Sum',output_const=1.,add_const=0.,sample_z=1,sample_W=1,W_sigma_prior=0.1, pooling_act='Sigmoid',BNN_init_range=0.01,
                 BNN_coef_sample=1.,KL_coef=1.,flag_local=False,couple_decoder_encoder=False,flag_log_q=False,flag_LV=False
               ):
        # Store argument
        self.latent_dim = latent_dim
        self.encoder_layer_num_before_agg = encoder_layer_num_before_agg
        self.encoder_layer_num_after_agg = encoder_layer_num_after_agg
        self.encoder_hidden_before_agg = encoder_hidden_before_agg
        self.encoder_hidden_after_agg = encoder_hidden_after_agg
        self.embedding_dim = embedding_dim
        self.obs_dim = obs_dim
        self.dim_before_agg = dim_before_agg
        self.decoder_layer_num = decoder_layer_num
        self.decoder_hidden = decoder_hidden
        self.output_const = output_const
        self.pooling = pooling
        self.sample_z = sample_z
        self.sample_W=sample_W
        self.W_sigma_prior=W_sigma_prior
        self.pooling_act=pooling_act
        self.flag_local=flag_local
        self.couple_decoder_encoder=False
        self.add_const=add_const
        self.flag_log_q=flag_log_q
        self.flag_LV=False
        # Default arguments
        self.flag_encoder_before_agg=True

        self.BNN_init_range=BNN_init_range
        self.BNN_coef_sample=BNN_coef_sample
        self.KL_coef=KL_coef
        ###################
        # Define Pooling function
        if pooling=='Sum':
            self.pool=torch.sum
        elif pooling=='Mean':
            self.pool=torch.mean
        elif pooling=='Max':
            pass
        else:
            raise NotImplemented
        if self.pooling_act=='ReLU':
            self.act_pool=F.relu
        elif self.pooling_act=='Sigmoid':
            self.act_pool=F.sigmoid
        elif self.pooling_act=='Tanh':
            self.act_pool=F.tanh
        elif type(self.pooling_act)==type(None):
            self.act_pool=None
        elif self.pooling_act=='Softplus':
            self.act_pool=F.softplus
        else:
            raise NotImplementedError
        self.encode_embedding,self.encode_bias=self._generate_default_embedding() # 1 x obs_dim x embed_dim and 1 x obs_dim x 1

        # BNN Decoder network

        if self.flag_local:
            self.decoder = BNN_last_local(self.latent_dim, self.obs_dim, hidden_layer_num=self.decoder_layer_num,
                                                hidden_unit=self.decoder_hidden,
                                                activations='ReLU', activations_output=None,
                                                flag_only_output_layer=False, sigma_prior=self.W_sigma_prior,
                                                init_range=self.BNN_init_range, coef_sample=self.BNN_coef_sample,flag_LV=self.flag_LV
                                                )
        else:
            raise NotImplementedError('flag_local must be True')

        if couple_decoder_encoder==True:
            raise NotImplementedError
        # Whether apply the transform before pooling function
        if type(self.encoder_layer_num_before_agg)==type(None):
            self.flag_encoder_before_agg=False
            self.dim_before_agg=self.embedding_dim
        else:
            self.encoder_before_agg = fc_Encoder_Decoder(self.embedding_dim + 2, output_dim=self.dim_before_agg,
                                                         hidden_layer_num=self.encoder_layer_num_before_agg,
                                                         hidden_unit=self.encoder_hidden_before_agg
                                                         , activations='ReLU', flag_only_output_layer=True,
                                                         activations_output='ReLU')
        # Apply after pooling
        self.encoder_after_agg = fc_Encoder_Decoder(self.dim_before_agg, output_dim=2 * self.latent_dim,
                                                    hidden_layer_num=self.encoder_layer_num_after_agg,
                                                    hidden_unit=self.encoder_hidden_after_agg,
                                                    activations='ReLU', flag_only_output_layer=False,
                                                    activations_output=None
                                                    )

    def _reset_last_layer_variance_BNN(self,sigma_range=0.01):
        self.decoder.out.W_sigma.data.uniform_(0,sigma_range)
        self.decoder.out.b_sigma.data.uniform_(0,sigma_range)

    def _generate_default_embedding(self):
        '''
        Generate the initial embeddings and bias with shape 1 x obs_dim x embedding_dim and 1 x obs_dim x 1
        :return: embedding,bias
        :rtype: Tensor, Tensor
        '''


        embedding=torch.randn(self.obs_dim,self.embedding_dim)
        embedding=torch.randn(1,self.embedding_dim)
        embedding=embedding.repeat(self.obs_dim,1)#+10*torch.randn(self.obs_dim,self.embedding_dim)

        embedding=torch.unsqueeze(embedding,dim=0) # 1 x obs_dim x embed_dim
        embedding=torch.tensor(embedding.data,requires_grad=True)

        bias=torch.randn(self.obs_dim,1)
        bias=torch.randn(1,1)
        bias=bias.repeat(self.obs_dim,1)
        bias=torch.unsqueeze(bias,dim=0) # 1 x obs_dim x 1
        bias=torch.tensor(bias.data,requires_grad=True)



        return embedding,bias
    def _encoding(self,X,mask):
        batch_size=X.shape[0] # N x obs_dim
        mask=torch.unsqueeze(mask,dim=len(mask.shape)) # N x obs x 1
        X_expand=torch.unsqueeze(X,dim=len(X.shape))  # N x obs x 1
        # Multiplicate embedding
        if len(self.encode_embedding.shape)!=len(X_expand.shape):
            encode_embedding=torch.unsqueeze(self.encode_embedding,dim=0)
        else:
            encode_embedding=self.encode_embedding
        X_embedding=X_expand*encode_embedding # N x obs x embed_dim
        if len(self.encode_bias.shape)!=len(X_embedding.shape):
            X_bias=torch.unsqueeze(self.encode_bias,dim=0) # 1 x 1 x obs x 1
            X_bias=X_bias.repeat(X_embedding.shape[0],X_embedding.shape[1],1,1) # N x d x obs x 1
        else:
            X_bias=self.encode_bias.repeat(batch_size,1,1) # N x obs x 1
        if self.couple_decoder_encoder==True:
            raise RuntimeError('couple_decoder_encoder must be False')
        if self.flag_encoder_before_agg==True:
            X_aug=torch.cat((X_expand,X_embedding,X_bias),dim=len(X_expand.shape)-1) # N x obs_dim x (embed+2)
            output_before_agg=1*self.encoder_before_agg.forward(X_aug) # N x obs x dim_before_agg
        else:
            output_before_agg=X_embedding+X_bias # N x obs x dim_before_agg

        mask_output_before_agg=mask*output_before_agg # N x obs x dim_before_agg

        if self.pooling != 'Max':
            agg_mask_output=self.pool(mask_output_before_agg, dim=len(mask_output_before_agg.shape)-2)
            if self.act_pool:
                agg_mask_output = self.act_pool(agg_mask_output)  # N x dim_before_agg
        elif self.pooling == 'Max':
            agg_mask_output=torch.max(mask_output_before_agg,dim=len(mask_output_before_agg.shape)-2)[0]
            if self.act_pool:
                agg_mask_output=self.act_pool(agg_mask_output)
        encoding=self.encoder_after_agg.forward(agg_mask_output)
        return encoding
    def sample_latent_variable(self,X,mask,size):
        batch_size=X.shape[0]
        encoding=self._encoding(X,mask)
        mean=encoding[:,:self.latent_dim]
        if self.flag_log_q==True:
            sigma=torch.sqrt(torch.exp(encoding[:, self.latent_dim:]))
        else:
            sigma = torch.sqrt((encoding[:, self.latent_dim:]) ** 2)

        #sigma=F.softplus(encoding[:,self.latent_dim:]) # N x latent_dim
        if size==1:
            eps=torch.randn(batch_size,self.latent_dim,requires_grad=True)
            z=mean+sigma*eps # N x latent_dim
        elif size>1:
            eps = torch.randn(size,batch_size, self.latent_dim, requires_grad=True)  # size x N  x latent_dim
            z = torch.unsqueeze(mean, dim=0) + torch.unsqueeze(sigma, dim=0) * eps  # size x N x latent_dim
        else:
            raise NotImplementedError
        return z,encoding

    def decoding(self,z,sigma_out,flag_record=True,size_W=None):

        if type(size_W) != type(None):
            size_W = size_W
        else:
            size_W = self.sample_W

        if self.flag_LV == False:

            mean = self.add_const + self.output_const * self.decoder.forward(z, num_sample=size_W,
                                                                             flag_record=flag_record)  # N_w x N x obs_dim or N_w x N_z x N x obs_dim
            sigma = sigma_out  # This is just a place holder. Not used in ELBO
            decode=mean+sigma*torch.randn(mean.shape)
            return decode, mean
        else:
            raise RuntimeError('flag_LV must be False')
    def target_ELBO(self,X,mask,W_sigma_prior,sigma_out,epoch,KL_coef_W,train_data_size,flag_BNN=True,coef_KL_coef_W=1.,flag_stream=False,target_dim=-1,**kwargs):

        # ASSUME THAT X CONTAINS THE TARGET VARIABLE, mask is mask for XUy
        # Exclude target variable from X
        XUy = torch.tensor(X.data)  # with target variable
        Xdy = torch.tensor(X.data)
        Xdy[:, target_dim] = 0.  # zero the target dim
        mask_Xdy = torch.tensor(mask)
        mask_Xdy[:, target_dim] = 0.

        latent_variable, encoding = self.sample_latent_variable(XUy, mask, size=self.sample_z)  # N x late_dim
        latent_variable_Xdy, encoding_Xdy = self.sample_latent_variable(Xdy, mask_Xdy,
                                                                        size=self.sample_z)  # N x late_dim
        q_mean = encoding[:, :self.latent_dim]
        q_mean_Xdy = torch.tensor(encoding_Xdy[:, :self.latent_dim].data)

        if self.flag_log_q == True:
            q_sigma = torch.sqrt(torch.exp(encoding[:, self.latent_dim:]))
            q_sigma_Xdy = torch.tensor(torch.sqrt(torch.exp(encoding_Xdy[:, self.latent_dim:])).data)
        else:
            q_sigma = torch.sqrt((encoding[:, self.latent_dim:]) ** 2)
            q_sigma_Xdy = torch.tensor(torch.sqrt((encoding_Xdy[:, self.latent_dim:]) ** 2).data)
        if flag_BNN == True:
            coef = coef_KL_coef_W * KL_coef_W(epoch)
        else:
            coef = 0.

        KL_z = self._KL_encoder_target_ELBO(q_mean, q_sigma, q_mean_Xdy, q_sigma_Xdy)  # N
        mask_y = torch.zeros(mask.shape)
        mask_y[:, target_dim] = 1.  # Only reserve the target dim and disable all other

        mask_LV = mask_y.clone().detach()
        if self.flag_LV == True:
            raise RuntimeError('flag_LV must be False')
        else:
            if flag_stream == False:
                KL_W = self._KL_decoder(mask, W_sigma_prior)  # N
            else:

                KL_W = self._KL_decoder_stream(mask, **kwargs)  # N


        mean_KL = torch.mean(KL_z + coef / train_data_size * KL_W)  # 1

        # Decode
        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')

        else:
            _, decoding = self.decoding(latent_variable, sigma_out,
                                                  flag_record=True)  # N_w x N_z x N x obs_dim or N_w x N x obs_dim
            # Log likelihood on y
            log_likelihood = self._gaussian_log_likelihood(XUy, mask_y, decoding,sigma_out)  # N_w x N_z x N or N_w x N

        if len(log_likelihood.shape) == 3:
            log_likelihood_N = torch.mean(torch.mean(log_likelihood, dim=0), dim=0)  # N
        elif len(log_likelihood.shape) == 2:
            log_likelihood_N = torch.mean(log_likelihood, dim=0)  # N
        mean_ELBO = torch.mean(log_likelihood_N) - 1. * mean_KL

        mean_ELBO_feature = 1. / (torch.sum(mask) + 0.001) * (torch.sum(log_likelihood_N))  # -KL_z-KL_W))

        return mean_ELBO, mean_ELBO_feature

    def ELBO(self,X,mask,W_sigma_prior,Z_sigma_prior,sigma_out,epoch,KL_coef_W,train_data_size,flag_BNN=True,coef_KL_coef_W=1.,flag_stream=False,**kwargs):

        active_dim_scale=torch.clamp(torch.sum(torch.abs(get_mask(X))>0.,dim=1).float(),min=1) # N
        active_dim=torch.clamp(torch.sum(torch.abs(mask)>0.,dim=1).float(),min=1) # N
        latent_variable, encoding = self.sample_latent_variable(X, mask, size=self.sample_z)
        q_mean = encoding[:, :self.latent_dim]

        # q_sigma=F.softplus(encoding[:,self.latent_dim:]) # N x latent_dim
        if self.flag_log_q == True:
            q_sigma = torch.sqrt(torch.exp(encoding[:, self.latent_dim:]))
        else:
            q_sigma = torch.sqrt((encoding[:, self.latent_dim:]) ** 2)

        if flag_BNN == True:
            # anneal the coef of KL penality of W. If use determinsitic PNP, coef=0.
            coef = coef_KL_coef_W * KL_coef_W(epoch)
        else:
            # This is for deterministic PNP
            coef = 0.

        KL_z = self._KL_encoder(q_mean, q_sigma, Z_sigma_prior)  # N
        mask_LV = torch.tensor(mask.data)
        if self.flag_LV == True:
            raise RuntimeError('flag_LV must be False')
        else:
            if flag_stream == False:

                KL_W = self._KL_decoder(mask, W_sigma_prior)

            else:

                KL_W = self._KL_decoder_stream(mask, **kwargs)

        mean_KL = torch.mean(KL_z*active_dim/active_dim_scale + coef / train_data_size * KL_W)  # 1


        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')

        else:
            _, decoding = self.decoding(latent_variable, sigma_out,
                                                  flag_record=True)  # N_w x N_z x N or N_w x N
            # log likelihood
            log_likelihood = self._gaussian_log_likelihood(X, mask, decoding,sigma_out)  # N_w x N_z x N or N_w x N


        if len(log_likelihood.shape) == 3:
            log_likelihood_N = torch.mean(torch.mean(log_likelihood, dim=0), dim=0)  # N
        elif len(log_likelihood.shape) == 2:
            log_likelihood_N = torch.mean(log_likelihood, dim=0)  # N

        mean_ELBO = torch.mean(log_likelihood_N) - self.KL_coef * mean_KL

        mean_ELBO_feature = 1. / (torch.sum(mask) + 0.001) * (torch.sum(log_likelihood_N))  # -KL_z-KL_W))
        return mean_ELBO, mean_ELBO_feature



    def _gaussian_log_likelihood(self,X,mask,decoding,sigma_out):
        X = X * mask
        decoding_size = len(decoding.shape)
        if decoding_size == 3:
            # N_w x N x obs_dim
            X_expand = torch.unsqueeze(X, dim=0)  # 1 x N x obs_dim
            mask_expand = torch.unsqueeze(mask, dim=0)  # 1 x N x obs_dim
            decoding = decoding * mask_expand  # N_w x N x obs_dim
            if self.flag_LV == False:
                log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                                                  dim=2) - 0.5 * torch.unsqueeze(torch.sum(mask, dim=1), dim=0) * (
                                             math.log(sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N
            else:
                raise RuntimeError('flag_LV must be False')
        elif decoding_size == 4:
            # N_w x N_z x N x obs
            X_expand = torch.unsqueeze(torch.unsqueeze(X, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            mask_expand = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            decoding = decoding * mask_expand  # N_w x N_z x N x obs_dim
            if self.flag_LV == False:
                log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                                                  dim=3) - 0.5 * torch.unsqueeze(
                    torch.unsqueeze(torch.sum(mask, dim=1), dim=0), dim=0) * (math.log(
                    sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N_z x N
            else:
                raise RuntimeError('flag_LV must be False')
        else:
            raise NotImplementedError
        return log_likelihood

    def _KL_decoder_stream(self,mask,**kwargs):
        # Stream training
        if self.decoder.flag_only_output_layer==True:
            pre_W_mean,pre_b_mean,pre_W_sigma,pre_b_sigma=kwargs['pre_W_mean'],kwargs['pre_b_mean'],kwargs['pre_W_sigma'],kwargs['pre_b_sigma']
            W_mean, b_mean, W_sigma, b_sigma = self.decoder._flatten_stat()  # N_in x N_out

            # Clamp the value
            pre_W_sigma, pre_b_sigma=torch.clamp(torch.abs(pre_W_sigma),min=0.0001,max=2.),torch.clamp(torch.abs(pre_b_sigma),min=0.0001,max=2.)
            W_sigma,b_sigma=torch.clamp(torch.abs(W_sigma),min=0.0001,max=2.),torch.clamp(torch.abs(b_sigma),min=0.0001,max=2.)

            mask_expand = torch.unsqueeze(mask, dim=1)  # N x 1 x obs_dim
            KL_mat_mean = torch.log(torch.abs(pre_W_sigma)/torch.abs(W_sigma))+0.5*(W_sigma**2)/(pre_W_sigma**2)+0.5*((W_mean-pre_W_mean)**2/(pre_W_sigma**2))-0.5 # N_in x N_out
            KL_mat_b= torch.log(torch.abs(pre_b_sigma)/torch.abs(b_sigma))+0.5*(b_sigma**2)/(pre_b_sigma**2)+0.5*((b_mean-pre_b_mean)**2/(pre_b_sigma**2))-0.5 # N_out

            KL_mat_mean_expand = torch.unsqueeze(KL_mat_mean, dim=0)
            KL_mat_b_expand = torch.unsqueeze(KL_mat_b, dim=0)  # 1 x obs

            KL_masked_mean = mask_expand * KL_mat_mean_expand  # N x n_in x n_out
            KL_masked_mean = torch.sum(torch.sum(KL_masked_mean, dim=1), dim=1)  # N

            KL_masked_b = torch.sum(mask * KL_mat_b_expand, dim=1)  # N
            KL_masked = KL_masked_b + KL_masked_mean

        else:
            raise NotImplementedError
        return KL_masked  # N
    def _KL_decoder(self,mask,W_sigma_prior):
        if self.decoder.flag_only_output_layer==True:
            #Only output layer
            W_mean, b_mean, W_sigma, b_sigma=self.decoder._flatten_stat() # N_in x N_out

            mask_expand = torch.unsqueeze(mask, dim=1)  # N x 1 x obs_dim

            KL_mat_mean = 0.5 * (math.log(W_sigma_prior ** 2) - torch.log(W_sigma ** 2) - 1 + W_sigma**2 / (W_sigma_prior**2) + (
                W_mean) ** 2 / (W_sigma_prior**2))  # N_in x obs_dim

            KL_mat_b = 0.5 * (math.log(W_sigma_prior ** 2) - torch.log(b_sigma ** 2) - 1 + b_sigma**2 / (W_sigma_prior**2) + (
                b_mean) ** 2 / (W_sigma_prior**2))  # obs_dim

            KL_mat_mean_expand = torch.unsqueeze(KL_mat_mean, dim=0)
            KL_mat_b_expand = torch.unsqueeze(KL_mat_b, dim=0)  # 1 x obs

            KL_masked_mean = mask_expand * KL_mat_mean_expand  # N x n_in x n_out
            KL_masked_mean = torch.sum(torch.sum(KL_masked_mean, dim=1), dim=1)  # N

            KL_masked_b = torch.sum(mask * KL_mat_b_expand, dim=1)  # N
            KL_masked = KL_masked_b + KL_masked_mean
        else:
            mean_flat, sigma_flat, W_mean, b_mean, W_sigma, b_sigma=self.decoder._flatten_stat()
            # For shared layer
            KL_shared=0.5*(torch.sum(math.log(W_sigma_prior**2)-torch.log(sigma_flat**2),dim=1)-sigma_flat.shape[1]+torch.sum((sigma_flat**2)/(W_sigma_prior**2),dim=1)+torch.sum((mean_flat**2)/(W_sigma_prior**2),dim=1))# 1

            # For output layer weight
            mask_expand = torch.unsqueeze(mask, dim=1)  # N x 1 x obs_dim

            KL_mat_mean = 0.5 * (math.log(W_sigma_prior ** 2) - torch.log(W_sigma ** 2) - 1 + (W_sigma**2) / (W_sigma_prior**2) + (
                W_mean) ** 2 / (W_sigma_prior**2))  # N_in x obs_dim

            KL_mat_b=0.5 * (math.log(W_sigma_prior ** 2) - torch.log(b_sigma ** 2) - 1 + (b_sigma**2) / (W_sigma_prior**2) + (
                b_mean) ** 2 / (W_sigma_prior**2)) # obs_dim


            KL_mat_mean_expand = torch.unsqueeze(KL_mat_mean, dim=0)
            KL_mat_b_expand=torch.unsqueeze(KL_mat_b,dim=0)  # 1 x obs

            KL_masked_mean = mask_expand * KL_mat_mean_expand  # N x n_in x n_out
            KL_masked_mean= torch.sum(torch.sum(KL_masked_mean, dim=1), dim=1)  # N

            KL_masked_b=torch.sum(mask*KL_mat_b_expand,dim=1) # N
            KL_masked=KL_masked_b+KL_masked_mean+KL_shared
        return KL_masked #N

    def _KL_encoder(self,mean,sigma,z_sigma_prior):
        KL_z=0.5*(math.log(z_sigma_prior**2)-torch.sum(torch.log(sigma**2),dim=1)-sigma.shape[1]+torch.sum(sigma**2/(z_sigma_prior**2),dim=1)+torch.sum(mean**2/(z_sigma_prior**2),dim=1)) #N
        return KL_z # N
    def _KL_encoder_target_ELBO(self,mean,sigma,mean_Xdy,sigma_Xdy):
        # clamp to avoid zero
        sigma=torch.clamp(torch.abs(sigma),min=0.001,max=10.)
        sigma_Xdy=torch.clamp(torch.abs(sigma_Xdy),min=0.001,max=10.)
        KL_z=torch.log(torch.abs(sigma_Xdy)/torch.abs(sigma))+0.5*(sigma**2)/(sigma_Xdy**2)+0.5*((mean-mean_Xdy)**2)/(sigma_Xdy**2)-0.5
        return torch.sum(KL_z,dim=1)
    def completion(self,X,mask,sigma_out,size=None):
        if type(size)!=type(None):
            size_z=size
            size_W=size
        else:
            size_z=self.sample_z
            size_W=self.sample_W
        z,_=self.sample_latent_variable(X,mask,size=size_z) # N x latent_dim or N_z x N x latent_dim

        if self.flag_LV:
            raise RuntimeError('flag_LV must be False')
        else:
            _,complete=self.decoding(z,sigma_out,flag_record=False,size_W=size_W) # N_w x N x obs_dim or N_w x N_z x N x obs_dim

        # if self.flag_sample==True:
        #     eps_sample=torch.randn(complete.shape)
        #     complete=complete+eps_sample*sigma_LV

        if len(complete.shape)==4:

            mean=torch.mean(torch.mean(complete,dim=0),dim=0) # N x obs_dim
            flat_complete=complete.view(-1,complete.shape[2],complete.shape[3])# (N_w x N_z) x N x obs_dim
            flat_complete=flat_complete-torch.unsqueeze(mean,dim=0) # (N_w x N_z) x N x obs_dim
            std_complete=torch.std(flat_complete,dim=0) # N x obs_dim
            var=std_complete**2+sigma_out**2


        elif len(complete.shape)==3:
            mean =torch.mean(complete, dim=0) # N x obs_dim
            flat_complete = complete  # N_w x N x obs_dim
            flat_complete = flat_complete - torch.unsqueeze(mean, dim=0)  # N_w x N x obs_dim
            std_complete = torch.std(flat_complete, dim=0)  # N x obs_dim
            var = std_complete ** 2 + sigma_out ** 2
        else:
            raise NotImplementedError
        return mean,var

    def test_log_likelihood(self,X_in,X_test,mask,sigma_out,size=None):

        if type(size)!=type(None):
            size_z=size
            size_W=size
        else:
            size_z=self.sample_z
            size_W=self.sample_W
        z, _ = self.sample_latent_variable(X_in, mask, size=size_z)  # N x latent_dim or N_z x N x latent_dim
        z=torch.tensor(z.data) # Clear memory
        if self.flag_LV:
            _, complete,sigma_LV = self.decoding(z, sigma_out, flag_record=False,size_W=size_W)  # N_w x N x obs_dim or N_w x N_z x N x obs_dim

        else:
            _, complete = self.decoding(z, sigma_out, flag_record=False,size_W=size_W)  # N_w x N x obs_dim or N_w x N_z x N x obs_dim

        complete=complete.clone().detach() # Clear memory
        test_mask=get_mask(X_test)
        if len(complete.shape) == 4:
            X_expand = torch.unsqueeze(torch.unsqueeze(X_test, dim=0), dim=0) # 1 x 1 x N x obs_dim
            mask_expand = torch.unsqueeze(torch.unsqueeze(test_mask, dim=0), dim=0)  # 1 x 1 x N x obs_dim
            decoding = complete * mask_expand  # N_w x N_z x N x obs_dim


            if self.flag_LV==True:
                raise RuntimeError('flag_LV must be False')
            else:
                log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                                                  dim=3) - 0.5 * torch.unsqueeze(
                    torch.unsqueeze(torch.sum(test_mask, dim=1), dim=0),
                    dim=0) * (math.log(
                    sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N_z x N


            log_likelihood=log_likelihood.view(-1,log_likelihood.shape[2]) # (N_w x N_z) x N
            pred_log_likelihood=torch.logsumexp(log_likelihood,dim=0)-math.log(float(log_likelihood.shape[0])) # N

            mean_pred_log_likelihood=1./(torch.sum(test_mask))*torch.sum(pred_log_likelihood)
            tot_pred_ll=torch.sum(pred_log_likelihood)
        elif len(complete.shape)==3:
            X_expand = torch.unsqueeze(X_test, dim=0)  #  1 x N x obs_dim
            mask_expand = torch.unsqueeze(test_mask, dim=0) #  1 x N x obs_dim
            decoding = complete * mask_expand  # N_w x N x obs_dim


            if self.flag_LV==True:
                raise RuntimeError('flag_LV must be False')
            else:
                log_likelihood = -0.5 * torch.sum((X_expand - decoding) ** 2 / (sigma_out ** 2),
                                                  dim=2) - 0.5 * torch.unsqueeze(torch.sum(test_mask, dim=1), dim=0) * (
                                             math.log(
                                                 sigma_out ** 2) + math.log(2 * np.pi))  # N_w x N

            pred_log_likelihood = torch.logsumexp(log_likelihood, dim=0) - math.log(float(log_likelihood.shape[0]))  # N

            mean_pred_log_likelihood = 1. / (torch.sum(test_mask)) * torch.sum(pred_log_likelihood)

            tot_pred_ll = torch.sum(pred_log_likelihood)
        else:
            raise NotImplementedError
        return torch.tensor(mean_pred_log_likelihood.data),torch.tensor(tot_pred_ll.data) # Clea

class Point_Net_Plus_BNN_SGHMC(object):
    def __init__(self,latent_dim,obs_dim,dim_before_agg,encoder_layer_num_before_agg=1,encoder_hidden_before_agg=None,encoder_layer_num_after_agg=1,
               encoder_hidden_after_agg=[100],embedding_dim=10,decoder_layer_num=2,decoder_hidden=[50,100],pooling='Sum',output_const=1.,add_const=0.,sample_z=1,sample_W=1,W_sigma_prior=0.1, pooling_act='Sigmoid',flag_log_q=False,flag_LV=False
               ):
        # Store argument
        self.latent_dim = latent_dim
        self.encoder_layer_num_before_agg = encoder_layer_num_before_agg
        self.encoder_layer_num_after_agg = encoder_layer_num_after_agg
        self.encoder_hidden_before_agg = encoder_hidden_before_agg
        self.encoder_hidden_after_agg = encoder_hidden_after_agg
        self.embedding_dim = embedding_dim
        self.obs_dim = obs_dim
        self.dim_before_agg = dim_before_agg
        self.decoder_layer_num = decoder_layer_num
        self.decoder_hidden = decoder_hidden
        self.output_const = output_const
        self.pooling = pooling
        self.sample_z = sample_z
        self.sample_W=sample_W
        self.W_sigma_prior=W_sigma_prior
        self.pooling_act=pooling_act
        self.add_const=add_const
        self.flag_log_q=flag_log_q
        self.flag_LV=False
        # Default arguments
        self.flag_encoder_before_agg=True

        # Define Pooling function
        if pooling=='Sum':
            self.pool=torch.sum
        elif pooling=='Mean':
            self.pool=torch.mean
        elif pooling=='Max':
            pass
        else:
            raise NotImplemented
        if self.pooling_act=='ReLU':
            self.act_pool=F.relu
        elif self.pooling_act=='Sigmoid':
            self.act_pool=F.sigmoid
        elif self.pooling_act=='Tanh':
            self.act_pool=F.tanh
        elif type(self.pooling_act)==type(None):
            self.act_pool=None
        elif self.pooling_act=='Softplus':
            self.act_pool=F.softplus
        else:
            raise NotImplementedError
        self.encode_embedding,self.encode_bias=self._generate_default_embedding() # 1 x obs_dim x embed_dim and 1 x obs_dim x 1

        # BNN Decoder network

        self.decoder=fc_Encoder_Decoder(self.latent_dim,self.obs_dim,hidden_layer_num=decoder_layer_num,hidden_unit=decoder_hidden,activations='ReLU',activations_output=None,flag_only_output_layer=False,drop_out_rate=0.,flag_drop_out=False,
                                        output_const=output_const,add_const=add_const,flag_LV=flag_LV)

        # Whether apply the transform before pooling function
        if type(self.encoder_layer_num_before_agg)==type(None):
            self.flag_encoder_before_agg=False
            self.dim_before_agg=self.embedding_dim
        else:
            self.encoder_before_agg = fc_Encoder_Decoder(self.embedding_dim + 2, output_dim=self.dim_before_agg,
                                                         hidden_layer_num=self.encoder_layer_num_before_agg,
                                                         hidden_unit=self.encoder_hidden_before_agg
                                                         , activations='ReLU', flag_only_output_layer=True,
                                                         activations_output='ReLU')
        # Apply after pooling
        self.encoder_after_agg = fc_Encoder_Decoder(self.dim_before_agg, output_dim=2 * self.latent_dim,
                                                    hidden_layer_num=self.encoder_layer_num_after_agg,
                                                    hidden_unit=self.encoder_hidden_after_agg,
                                                    activations='ReLU', flag_only_output_layer=False,
                                                    activations_output=None
                                                    )
    def _generate_default_embedding(self):
        '''
        Generate the initial embeddings and bias with shape 1 x obs_dim x embedding_dim and 1 x obs_dim x 1
        :return: embedding,bias
        :rtype: Tensor, Tensor
        '''


        embedding=torch.randn(self.obs_dim,self.embedding_dim)
        #embedding=torch.randn(1,self.embedding_dim)
        #embedding=embedding.repeat(self.obs_dim,1)#+10*torch.randn(self.obs_dim,self.embedding_dim)

        embedding=torch.unsqueeze(embedding,dim=0) # 1 x obs_dim x embed_dim
        embedding=torch.tensor(embedding.data,requires_grad=True)

        bias=torch.randn(self.obs_dim,1)
        #bias=torch.randn(1,1)
        #bias=bias.repeat(self.obs_dim,1)
        bias=torch.unsqueeze(bias,dim=0) # 1 x obs_dim x 1
        bias=torch.tensor(bias.data,requires_grad=True)
        return embedding,bias
    def _encoding(self,X,mask):
        batch_size=X.shape[0] # N x obs_dim
        mask=torch.unsqueeze(mask,dim=len(mask.shape)) # N x obs x 1
        X_expand=torch.unsqueeze(X,dim=len(X.shape))  # N x obs x 1
        # Multiplicate embedding
        if len(self.encode_embedding.shape)!=len(X_expand.shape):
            encode_embedding=torch.unsqueeze(self.encode_embedding,dim=0)
        else:
            encode_embedding=self.encode_embedding
        X_embedding=X_expand*encode_embedding # N x obs x embed_dim

        if len(self.encode_bias.shape)!=len(X_embedding.shape):
            X_bias=self.encode_bias.expand_as(X_embedding)
            X_bias=torch.index_select(X_bias,dim=-1,index=torch.tensor([0]))
        else:
            X_bias=self.encode_bias.repeat(batch_size,1,1) # N x obs x 1

        if self.flag_encoder_before_agg==True:
            X_aug=torch.cat((X_expand,X_embedding,X_bias),dim=len(X_expand.shape)-1) # N x obs_dim x (embed+2)
            output_before_agg=1*self.encoder_before_agg.forward(X_aug) # N x obs x dim_before_agg
        else:
            output_before_agg=X_embedding+X_bias # N x obs x dim_before_agg

        mask_output_before_agg=mask*output_before_agg # N x obs x dim_before_agg

        if self.pooling != 'Max':
            agg_mask_output=self.pool(mask_output_before_agg, dim=len(mask_output_before_agg.shape)-2)
            if self.act_pool:
                agg_mask_output = self.act_pool(agg_mask_output)  # N x dim_before_agg
        elif self.pooling == 'Max':
            agg_mask_output=torch.max(mask_output_before_agg,dim=len(mask_output_before_agg.shape)-2)[0]
            if self.act_pool:
                agg_mask_output=self.act_pool(agg_mask_output)
        encoding=self.encoder_after_agg.forward(agg_mask_output)
        # Encoding -3
        encoding[:,self.latent_dim:]=encoding[:,self.latent_dim:]-0.
        return encoding

    def sample_latent_variable(self,X,mask,size=10):
        batch_size = X.shape[0]
        encoding = self._encoding(X, mask)
        mean = encoding[:, :self.latent_dim]
        if self.flag_log_q == True:
            sigma = torch.clamp(torch.sqrt(torch.exp(encoding[:, self.latent_dim:])),min=0,max=300.)
        else:
            sigma = torch.clamp(torch.sqrt((encoding[:, self.latent_dim:]) ** 2),min=0.,max=300.)
        if size == 1:
            eps = torch.randn(batch_size, self.latent_dim, requires_grad=True)
            z = mean + sigma * eps  # N x latent_dim
        elif size > 1:
            eps = torch.randn(size, batch_size, self.latent_dim, requires_grad=True)  # size x N  x latent_dim
            z = torch.unsqueeze(mean, dim=0) + torch.unsqueeze(sigma, dim=0) * eps  # size x N x latent_dim
        else:
            raise NotImplementedError
        return z, encoding

    def _KL_encoder(self,mean,sigma,z_sigma_prior=1.):
        KL_z = 0.5 * (
                    math.log(z_sigma_prior ** 2) - torch.sum(torch.log(sigma ** 2), dim=1) - sigma.shape[1] + torch.sum(
                sigma ** 2 / (z_sigma_prior ** 2), dim=1) + torch.sum(mean ** 2 / (z_sigma_prior ** 2), dim=1))  # N

        return KL_z  # N
    def test_log_likelihood(self,Infer_model,X_in,X_test,W_sample,mask,sigma_out,size=10):
        mean_pred_log_likelihood,tot_pred_ll=Infer_model.test_log_likelihood(X_in, X_test, W_sample, mask, sigma_out, size=size)
        return mean_pred_log_likelihood,tot_pred_ll
    def completion(self,Infer_model,X,mask,W_sample,size_Z=10,record_memory=False):
        complete=Infer_model.completion(X,mask,W_sample,size_Z=size_Z,record_memory=False)
        return complete
    def _KL_encoder_target_ELBO(self,q_mean_XUy, q_sigma_XUy, q_mean_Xdy, q_sigma_Xdy):
        # clamp to avoid zero
        sigma = torch.clamp(torch.abs(q_sigma_XUy), min=0.001, max=10.)
        sigma_Xdy = torch.clamp(torch.abs(q_sigma_Xdy), min=0.001, max=10.)
        KL_z = torch.log(torch.abs(sigma_Xdy) / torch.abs(sigma)) + 0.5 * (sigma ** 2) / (sigma_Xdy ** 2) + 0.5 * (
                    (q_mean_XUy - q_mean_Xdy) ** 2) / (sigma_Xdy ** 2) - 0.5 # N x latent_dim
        return torch.sum(KL_z, dim=1)

    def decoding(self,Infer_model,z,W_sample):
        X=Infer_model.sample_X(z,W_sample)
        return X