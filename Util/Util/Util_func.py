import torch
import numpy as np
from Dataloader.base_mask import *
import copy
import pickle
import yaml
import math
from prettytable import PrettyTable
#from Dataloader.base_Dataloader import base_UCI_Dataset
from torch.utils.data import DataLoader
from scipy.stats import bernoulli
import collections
import re
import torch.nn.functional as F

#from cachetools import LRUCache

# Update W_sample
def Update_W_sample(W_dict,W_sample,sample_num,maxsize=20):
    if len(W_sample)>=maxsize:
        key,value=W_sample.popitem(last=False)
        #print('%s is evicted'%(key))
        W_sample['sample_%s'%(sample_num)]=copy.deepcopy(W_dict)
    else:
        W_sample['sample_%s' % (sample_num)] = copy.deepcopy(W_dict)
    return W_sample



class base_UCI_Dataset(Dataset):
    '''
    Most simple dataset by explicit giving train and test data
    '''
    def __init__(self,data,transform=None,flag_GPU=True):
        self.Data=data
        self.flag_GPU=flag_GPU
        self.transform=transform
    def __len__(self):
        return self.Data.shape[0]
    def __getitem__(self, idx):
        sample=self.Data[idx,:]
        if self.transform and self.flag_GPU==True:
            sample=self.transform(sample)
            sample=sample.cuda()
        elif self.transform and not self.flag_GPU:
            sample=self.transform(sample)
        return sample

def train_PNP(model,dataset,epoch_Ref,conditional_coef_Ref=0.75,flag_hybrid=True,**kwargs):
    batch_size = kwargs['batch_size']
    z_sigma_prior = kwargs['z_sigma_prior']
    W_sigma_prior = kwargs['W_sigma_prior']

    sigma_out = kwargs['sigma_out']
    Drop_p=0.2
    train_data = dataset
    coef_KL_coef_W = 1.
    train_data_size = get_train_data_size(train_data)
    Optim_settings=kwargs['Optim_Settings']
    test_input_tensor=kwargs['test_input_tensor']
    test_target_tensor=kwargs['test_target_tensor']
    # Define optimizer
    Adam_encoder = torch.optim.Adam(
        list(model.encoder_before_agg.parameters()) + list(model.encoder_after_agg.parameters()),
        lr=Optim_settings['lr'],
        betas=(Optim_settings['beta1'], Optim_settings['beta2']),
        weight_decay=Optim_settings['weight_decay'])
    Adam_decoder = torch.optim.Adam(
        list(model.decoder.parameters()), lr=Optim_settings['lr'],
        betas=(Optim_settings['beta1'], Optim_settings['beta2']),
        weight_decay=Optim_settings['weight_decay'])
    Adam_embedding = torch.optim.Adam(
        [model.encode_embedding, model.encode_bias], lr=Optim_settings['lr'],
        betas=(Optim_settings['beta1'], Optim_settings['beta2']),
        weight_decay=Optim_settings['weight_decay'])
    flag_BNN = kwargs['flag_BNN']
    KL_coef_W = kwargs['KL_coef_W']
    train_dataset = base_UCI_Dataset(train_data, transform=None, flag_GPU=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    acc_ELBO_feature = 0.

    RMSE_mat, MAE_mat, NLL_mat = np.zeros(epoch_Ref), np.zeros(epoch_Ref), np.zeros(epoch_Ref)
    for ep in range(epoch_Ref):
        if (ep + 1) % 100 == 0:
            print('PNP_BNN Epoch:%s/%s' % (ep + 1, epoch_Ref))
        for idx, data in enumerate(train_loader):
            # zero grad
            Adam_encoder.zero_grad()
            Adam_decoder.zero_grad()
            Adam_embedding.zero_grad()
            # Generate drop mask
            mask = get_mask(data)
            # Debug variable artificial missingness
            Drop_p_var = np.minimum(np.random.rand(1), Drop_p)
            mask_drop = np.array([bernoulli.rvs(1 - Drop_p_var, size=data.shape[1])] * data.shape[0])
            mask_drop = torch.from_numpy(mask_drop).float().cuda()
            if flag_hybrid:
                mask_drop_hybrid = torch.tensor(mask_drop.data)
                mask_drop_hybrid[:, -1] = 1.  # Reserve the target dim
                mask_drop_Xdy = torch.tensor(mask_drop.data)
                mask_drop_Xdy[:, -1] = 0.
                mask_target_ELBO = mask * mask_drop_hybrid
                mask = mask * mask_drop
            else:
                mask = mask * mask_drop

            elbo_mean, elbo_mean_feature = model.ELBO(data, mask, W_sigma_prior, z_sigma_prior, sigma_out,
                                                               epoch=ep, KL_coef_W=KL_coef_W,
                                                               train_data_size=train_data_size,
                                                               flag_BNN=flag_BNN, flag_stream=False,
                                                               coef_KL_coef_W=coef_KL_coef_W)


            if flag_hybrid:  # Enable conditional training

                elbo_mean_target, elbo_mean_feature_target = model.target_ELBO(data, mask_target_ELBO,
                                                                                        W_sigma_prior,
                                                                                        sigma_out,
                                                                                        epoch=ep,
                                                                                        KL_coef_W=KL_coef_W,
                                                                                        train_data_size=train_data_size,
                                                                                        flag_BNN=flag_BNN,
                                                                                        flag_stream=False,
                                                                                        coef_KL_coef_W=coef_KL_coef_W,
                                                                                        target_dim=-1)


                elbo_mean = conditional_coef_Ref * elbo_mean + (1. - conditional_coef_Ref) * elbo_mean_target
                elbo_feature_target = elbo_mean_feature_target  # conditional_coef*elbo_mean_feature+(1.-conditional_coef)*elbo_mean_feature_target

            acc_ELBO_feature = acc_ELBO_feature + elbo_feature_target.data.cpu().numpy()
            loss = -elbo_mean
            loss.backward()
            # Update Parameters
            Adam_encoder.step()
            Adam_decoder.step()
            Adam_embedding.step()
        if (ep + 1) % 100 == 0:
            print('Training loss:%s' % (acc_ELBO_feature / 100.))
            acc_ELBO_feature = 0.
        # For DEBUG
        if (ep + 1) % 1 == 0:
            RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model, test_input_tensor,
                                                           test_target_tensor,
                                                           sigma_out_scale=sigma_out,
                                                           split=10, flag_model='PNP_BNN')
            RMSE_mat[ep], MAE_mat[ep], NLL_mat[ep] = RMSE_test.cpu().data.numpy(), MAE_test.cpu().data.numpy(), NLL_test.cpu().data.numpy()

            if (ep+1)%100==0:
                print('ep:%s NLL:%s RMSE:%s' % (ep+1,
                NLL_test.cpu().data.numpy(),RMSE_test.cpu().data.numpy()))
    return RMSE_mat,MAE_mat,NLL_mat



def test_UCI_AL_Ref(model,Ref_model,max_selection,sample_x,test_input,test_pool,test_target,sigma_out,search='Target'):
    RMSE_Results=np.zeros(max_selection+1)
    MAE_Results=np.zeros(max_selection+1)
    NLL_Results=np.zeros(max_selection+1)

    # Evaluate on zero selection

    RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                   sigma_out_scale=sigma_out,
                                                   split=10, flag_model='PNP_BNN', size=25)
    RMSE_Results[0] = RMSE_test.cpu().data.numpy()
    MAE_Results[0] = MAE_test.cpu().data.numpy()
    NLL_Results[0] = NLL_test.cpu().data.numpy()
    for num_selected in range(max_selection):
        # Active Learning
        if search=='Target':
            test_input, index_array, test_pool = model.base_active_learning_z_target_BNN(
                active_sample_number=sample_x,
                test_input=test_input,
                pool_data_tensor=test_pool, target_data_tensor=test_target)
        elif search=='Random':
            test_input, index_array, test_pool = model.base_random_select(
                active_sample_number=sample_x, test_input=test_input, pool_data_tensor=test_pool)
        else:
            raise NotImplementedError
        # Clear memory
        test_input=torch.tensor(test_input.data)
        test_pool=torch.tensor(test_pool.data)

        RMSE_test, MAE_test, NLL_test = Test_UCI_batch(Ref_model, test_input, test_target,
                                                                      sigma_out_scale=sigma_out,
                                                                      split=10, flag_model='PNP_BNN', size=25)
        RMSE_Results[num_selected+1]=RMSE_test.cpu().data.numpy()
        MAE_Results[num_selected+1]=MAE_test.cpu().data.numpy()
        NLL_Results[num_selected+1]=NLL_test.cpu().data.numpy()
    return RMSE_Results,MAE_Results,NLL_Results

def Test_RMSE(data_loader,PNP,sigma_out,mask_prop):
    RMSE_tot=0
    for idx,data in enumerate(data_loader):
        mask = base_generate_mask(data.shape[0], data.shape[1], mask_prop=mask_prop)
        complete_data=PNP.completion(data,mask,sigma_out)
        RMSE=torch.sum(torch.norm(data-complete_data,dim=1))
        RMSE_tot+=RMSE
    RMSE_tot=1./len(data_loader.dataset)*RMSE_tot
    return RMSE_tot

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def Test_UCI_batch(PNP,test_input,test_target,sigma_out_scale=0.1,split=3,flag_model='PNP_BNN',size=10,Infer_model=None,W_sample=None):
    batch_size = int(math.ceil(test_input.shape[0] / split))
    pre_idx = 0
    PNP_copy = PNP#copy.deepcopy(PNP)
    total_se = 0
    total_MAE = 0
    total_NLL =0
    for counter in range(split+1):
        idx=min((counter+1)*batch_size,test_input.shape[0])
        if pre_idx==idx:
            break
        data_input=test_input[pre_idx:idx,:]
        data_target=test_target[pre_idx:idx,:]
        # get mask
        mask_test_input = get_mask(data_input)
        mask_test_target = get_mask(data_target)
        if flag_model=='PNP_BNN':
            pred_mean,_=PNP_copy.completion(data_input,mask_test_input,sigma_out_scale,size=size)
            pred_mean=pred_mean*mask_test_target
            _,pred_tot_ll=PNP_copy.test_log_likelihood(data_input,data_target,mask_test_input,sigma_out_scale,size=size)
        elif flag_model=='PNP':
            sigma_out = sigma_out_scale * torch.ones(1, test_input.shape[1])
            pred_mean = PNP_copy.completion(data_input, mask_test_input, sigma_out) * mask_test_target
            _,pred_tot_ll=PNP_copy.test_log_likelihood(data_input,data_target,mask_test_input,sigma_out)
        elif flag_model=='PNP_SGHMC':
            if Infer_model.flag_LV:
                pred_mean,_ = PNP_copy.completion(Infer_model,data_input, mask_test_input,W_sample, size_Z=size,record_memory=False)

            else:
                pred_mean = PNP_copy.completion(Infer_model,data_input, mask_test_input,W_sample, size_Z=size,record_memory=False)
            pred_mean=torch.mean(torch.mean(pred_mean,dim=0),dim=0)
            pred_mean = pred_mean * mask_test_target


            _, pred_tot_ll = PNP_copy.test_log_likelihood(Infer_model,X_in=data_input, X_test=data_target,W_sample=W_sample, mask=mask_test_input,sigma_out=sigma_out_scale,
                                                          size=size)
        else:
            raise NotImplementedError

        pred_test = torch.tensor(pred_mean.data)
        ae = torch.sum(torch.abs(pred_test - data_target))
        ae = torch.tensor(ae.data)
        se = torch.sum((pred_test - data_target) ** 2)
        se = torch.tensor(se.data)
        total_se += se
        total_MAE += ae
        total_NLL+=-pred_tot_ll
        pre_idx = idx
    total_mask_target = get_mask(test_target)
    RMSE = torch.sqrt(1. / (torch.sum(total_mask_target)) * total_se)
    RMSE = torch.tensor(RMSE.data)
    MAE = torch.tensor((1. / torch.sum(total_mask_target) * total_MAE).data)
    NLL=torch.tensor(1. / (torch.sum(total_mask_target)) *total_NLL)
    return RMSE, MAE, NLL

def test_UCI_AL(model,max_selection,sample_x,test_input,test_pool,test_target,sigma_out,search='Target',model_name='PNP_BNN',**kwargs):
    RMSE_Results=np.zeros(max_selection+1)
    MAE_Results=np.zeros(max_selection+1)
    NLL_Results=np.zeros(max_selection+1)
    if model_name=='PNP_SGHMC':
        W_sample=kwargs['W_sample']
        Infer_model=model.Infer_model
    # Evaluate on zero selection

        RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                   sigma_out_scale=sigma_out,
                                                   split=10, flag_model=model_name, size=25,Infer_model=Infer_model,W_sample=W_sample)
    else:
        RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                       sigma_out_scale=sigma_out,
                                                       split=10, flag_model=model_name, size=25)

    RMSE_Results[0] = RMSE_test.cpu().data.numpy()
    MAE_Results[0] = MAE_test.cpu().data.numpy()
    NLL_Results[0] = NLL_test.cpu().data.numpy()
    for num_selected in range(max_selection):
        # Active Learning
        if search=='Target' and num_selected>-1:
            if model_name=='PNP_BNN':
                test_input, index_array, test_pool = model.base_active_learning_z_target_BNN(
                    active_sample_number=sample_x,
                    test_input=test_input,
                    pool_data_tensor=test_pool, target_data_tensor=test_target)
            elif model_name=='PNP_SGHMC':
                test_input, index_array, test_pool = model.active_learn_target_test(flag_same_pool=True,test_input=test_input,pool_data_tensor=test_pool,target_data_tensor=test_target,size_z=10,split=10,W_sample=W_sample)
            else:
                raise NotImplementedError
        elif search=='Random' :
            if model_name=='PNP_BNN':
                test_input, index_array, test_pool = model.base_random_select(
                    active_sample_number=sample_x, test_input=test_input, pool_data_tensor=test_pool)
            elif model_name=='PNP_SGHMC':
                test_input, index_array, test_pool = model.random_select_test( test_input=test_input,pool_data_tensor=test_pool)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # Clear memory
        test_input=torch.tensor(test_input.data)
        test_pool=torch.tensor(test_pool.data)
        if model_name=='PNP_SGHMC':
            RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                                          sigma_out_scale=sigma_out,
                                                                          split=10, flag_model=model_name, size=25,Infer_model=Infer_model,W_sample=W_sample)
        else:
            RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                           sigma_out_scale=sigma_out,
                                                           split=10, flag_model=model_name, size=25)
        RMSE_Results[num_selected+1]=RMSE_test.cpu().data.numpy()
        MAE_Results[num_selected+1]=MAE_test.cpu().data.numpy()
        NLL_Results[num_selected+1]=NLL_test.cpu().data.numpy()
    return RMSE_Results,MAE_Results,NLL_Results









def Movie_get_temperature(overall_data,max_rating=5):
    '''
    Get the weight term for likelihood term
    :param overall_data:
    :type overall_data:
    :return:
    :rtype:

    '''
    freq_rating=np.zeros(max_rating)
    for idx in range(max_rating):
        rating=idx+1
        num_rating=np.sum(overall_data==rating)
        freq_rating[idx]=num_rating
    maxi_rat=np.max(freq_rating)

    weight_array=maxi_rat/freq_rating
    weight_array=np.array([1.,1.,1.,1.,1.])
    return weight_array
def Movie_generate_temp_mat(X,weight_array):
    '''
    :param X:
    :type X:
    :param weight_array:
    :type weight_array:
    :return:
    :rtype:
    '''
    weight_mat=torch.zeros(X.shape)
    max_rat=torch.max(X).cpu().data.numpy()
    for idx in range(int(max_rat)):
        rat=idx+1
        weight_mat[X==rat]=weight_array[idx]
    return weight_mat
def save_data(filename,data):
    savefile = open(filename, 'wb')
    pickle.dump(data, savefile)
    savefile.close()

class Results_Class(object):
    def __init__(self,**kwargs):
        self.Data=kwargs
    def _list_of_keys(self):
        key_list=[]
        for key,_ in self.Data.items():
            key_list.append(key)
        return key_list
    def std(self,key):
        raise NotImplementedError
def load_data(File_name):
    pkl_file = open(File_name, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data
def remove_ratings(X,rating=1):
    assert type(rating)==int,'Rating must be integer'
    X[X==rating]=0
    return torch.tensor(X.data)


def ReadYAML(Filename):
    with open(Filename,'r') as ymlfile:
        cfg=yaml.load(ymlfile)
    return cfg
def KL_Schedule(ep,**kwargs):
    schedule=kwargs['schedule']
    if schedule =='0':
        # const coef
        coef=kwargs['coef']
        return coef
    elif schedule =='1':
        # Linear Warm-up from 0 to 1
        epoch_target=kwargs['epoch_target']
        if ep<epoch_target:
            coef=ep*kwargs['coef']/(epoch_target)
        else:
            coef=kwargs['coef']
        return coef
    elif schedule=='2':
        # Exponential decay
        epoch_target=kwargs['epoch_target']
        coef=0.001 + math.exp(-(ep) / (epoch_target))
        return coef
    else:
        raise NotImplementedError
def KL_Schedule_helper(**kwargs):
    return lambda ep: KL_Schedule(ep,**kwargs)
def get_train_data_size(train_data):
    sum_train_data=torch.sum(torch.abs(train_data),dim=1)
    train_size=float(torch.sum(sum_train_data>0.).float().data.cpu().numpy())
    return train_size
def get_correct_scale(result_type,extra_range,remove_last=True,dim=None,**kwargs):
    # For plot with correct scales
    max_list=np.zeros(len(kwargs))
    min_list=np.zeros(len(kwargs))
    counter=0
    for key,value in kwargs.items():
        std_error=np.sqrt(float(value.Data[result_type].shape[0]))
        if remove_last==True:
            results=value.Data[result_type][:,0:-1]
        else:
            results=value.Data[result_type]
        average_results=np.mean(results,axis=0)
        std_results=np.std(results,axis=0)/std_error
        max_candidate=average_results+std_results
        min_candidate=average_results-std_results
        if type(dim)!=None:
            max_results = np.max(max_candidate[:,dim])
            min_results = np.min(min_candidate[:,dim])
            max_list[counter] = max_results
            min_list[counter] = min_results
        else:
            max_results=np.max(max_candidate)
            min_results=np.min(min_candidate)
            max_list[counter]=max_results
            min_list[counter]=min_results
        counter+=1
    max_all=np.max(max_list)
    min_all=np.min(min_list)
    max_min_range=(max_all-min_all)*extra_range
    scale_up=max_all+max_min_range
    scale_down=min_all-max_min_range
    return scale_up,scale_down
def get_choice(observed_train_data):
    num_selected=torch.sum(torch.abs(observed_train_data)>0.,dim=0) # obs_dim
    return num_selected.cpu().data.numpy()

########### They are for post-process the results matrix
def remove_continuous_zero_column(data):
    sum_data=np.sum(np.abs(data),axis=0)
    idx_zero=np.where(sum_data<0.01)[0][0]
    return data[:,:idx_zero]
def remove_continuous_zero_choice(choice):
    sum_choice=np.sum(np.sum(choice,axis=0),axis=1) # 1000
    idx_zero=np.where(np.abs(sum_choice)<0.01)[0][1]
    return choice[:,:idx_zero,:]
def squeeze_range_1D(data,range):
    assert len(data.shape)==1,'Only 1D array is supported'
    new_data=np.interp(data,(data.min(),data.max()),(range[0],range[1]))
    return new_data

#########################

def remove_zero_row_2D(data):
    sum_data=torch.sum(torch.abs(data),dim=1) # N
    return torch.tensor(data[sum_data>0,:].data)

def assign_zero_row_2D_with_target(data,data_target,value=0.):
    data=torch.tensor(data.data)
    sum_data=torch.sum(torch.abs(data_target),dim=1)
    data[sum_data<=0.001,:]=value
    return data # Can be seen as the entries of data matrix with the positions of the unobserved rows of the data_target are 0
def Compute_AUIC_AUIC(const=None,**kwargs):
    # For Debug Prupose
    #TODO: Check if this is used in other files
    if type(const)==type(None):
        Result1=kwargs['Result1'] # runs x selected_points
        Result2=kwargs['Result2']
        diff=Result1-Result2
        diff_roll = np.roll(diff, -1, axis=1)
        # Compute AUIC
        area = np.sum((0.5 * (diff + diff_roll))[:, 0:-1], axis=1) # runs
        return np.mean(area),np.std(area)
    else:
        Result1 = kwargs['Result1']  # runs x selected_points
        diff = Result1 - const
        diff_roll = np.roll(diff, -1, axis=1)
        # Compute AUIC
        area = np.sum((0.5 * (diff + diff_roll))[:, 0:-1], axis=1)  # runs
        return np.mean(area), np.std(area)

def Compute_AUIC(const=None,flag_variance=False,**kwargs):
    if type(const)==type(None): # Compute Area difference
        BALD_Results=kwargs['BALD']
        RAND_Results=kwargs['RAND'] # Runs x counter_loop x selection_variable

        if flag_variance==False:
            mean_BALD=np.mean(BALD_Results,axis=0) # counter_loop x selection_variable
            mean_RAND=np.mean(RAND_Results,axis=0)
        else:
            mean_BALD=BALD_Results
            mean_RAND=RAND_Results

        diff=mean_RAND-mean_BALD # counter_loop x selection_variable
        # compute the sum
        if flag_variance==False:
            diff_roll=np.roll(diff,-1,axis=1)
            # Compute AUIC
            area = np.sum((0.5 * (diff + diff_roll))[:, 0:-1], axis=1)
        else:
            diff_roll = np.roll(diff, -1, axis=2)
            # Compute AUIC
            area = np.sum((0.5 * (diff + diff_roll))[:, :,0:-1], axis=2)

        return area
    else:
        Results=kwargs['Results']
        if flag_variance==False:
            mean_Results=np.mean(Results,axis=0)
        else:
            mean_Results=Results
        diff=mean_Results-const
        if flag_variance==False:
            diff_roll=np.roll(diff,-1,axis=1)
            area=np.sum((0.5*(diff+diff_roll))[:,0:-1],axis=1)
        else:
            diff_roll = np.roll(diff, -1, axis=2)
            area = np.sum((0.5 * (diff + diff_roll))[:, :,0:-1], axis=2)
        return area
def Compute_AUIC_1D(const=None,**kwargs):

    if type(const)==type(None):
        BALD = kwargs['BALD']
        RAND = kwargs['RAND']
        # Convert to np
        #BALD = BALD.data.cpu().numpy()
        #RAND = RAND.data.cpu().numpy()
        diff=RAND-BALD
        diff_roll=np.roll(diff,-1)
        area = np.sum((0.5 * (diff + diff_roll))[0:-1], axis=0)
    else:
        Results=kwargs['Results']
        #Results=Results.cpu().data.numpy()
        diff=Results-const
        diff_roll=np.roll(diff,-1)
        area = np.sum((0.5 * (diff + diff_roll))[0:-1], axis=0)
    return area
def Remove_RandSeedResults(dim,Data):
    # This is to remove the outlier results

    for key_in,value_in in Data.items():
        Data[key_in]=np.delete(Data[key_in],dim,axis=0) # Remove the outlier
    return Data
def Remove_AUIC(idx,AUIC):
    AUIC_new=np.delete(AUIC,idx,axis=0)
    return AUIC_new
def BALD_Select_Explore(BALD_weight,step):
    # THis implement the selection basded on the probability defined by BALD value
    active_entry_num=torch.sum(BALD_weight>0.)

    if active_entry_num==0.: # Every thing is zero, means no way to draw it
        flag_full = True
        idx = []
        select_num=0
    elif active_entry_num<step:
        flag_full=False
        idx=torch.multinomial(BALD_weight,active_entry_num)
        select_num=active_entry_num
    else:
        flag_full = False
        idx = torch.multinomial(BALD_weight, step)
        select_num=step
    return idx,flag_full,select_num
def BALD_Select(BALD):
    BALD=torch.tensor(BALD.data)
    num_unobserved=torch.sum(BALD<-10)

    # Active Select
    _, idx = torch.sort(BALD.view(1, -1), descending=True)  # Flattened idx # 1 x tot
    return idx,num_unobserved
def BALD_Select_Explore_batch(BALD,idx_start,idx_end,pool_mask_byte,temp_scale):
    # Note: This should be pure BALD value, without softmax and temperature
    BALD=BALD.clone().detach()
    if idx_end>BALD.shape[0]:
        idx_end=BALD.shape[0]
    BALD_selected=BALD[idx_start:idx_end,:]

    pool_mask_byte_selected=pool_mask_byte[idx_start:idx_end,:]
    BALD_selected[pool_mask_byte_selected]=-100.
    for user_id in range(idx_end-idx_start):
        BALD_current = BALD_selected[user_id, :]
        # Active index
        active_num_current=torch.sum(BALD_current>-100.)
        if active_num_current>0:
            # Tempering for the current user

            mean_scale = torch.mean(torch.abs(BALD_current[BALD_current > -100.]))
            temp = (temp_scale + 0.) * mean_scale

            BALD_current[BALD_current > -100.] = torch.clamp(F.softmax(BALD_current[BALD_current > -100.] / temp), max=1.,
                                                                min=1e-10) # obs_dim
            BALD_current[BALD_current<0.]=0.
            idx_current = torch.multinomial(BALD_current, 1)
        else:
            idx_current=torch.tensor([-100]).long()
        if user_id==0:
            idx=idx_current.clone().detach()
        else:
            idx=torch.cat((idx,idx_current),dim=0)
    num_selected=torch.sum(idx>-100)
    return idx,num_selected
def Random_Select_batch(pool_data_cp,idx_start,idx_end):
    if idx_end > pool_data_cp.shape[0]:
        idx_end=pool_data_cp.shape[0]
    pool_data_cp_selected=pool_data_cp[idx_start:idx_end,:]
    for user_id in range(idx_end-idx_start):
        pool_data_cp_current=pool_data_cp_selected[user_id,:]
        active_num_current = torch.sum(torch.abs(pool_data_cp_current)>0.)
        if active_num_current>0:
            idx_obs = (torch.abs(pool_data_cp_current) > 0.).nonzero()
            idx_obs = idx_obs[:, 0]


            idx_random = torch.randperm(idx_obs.shape[0])

            idx_selected = idx_obs[idx_random]
            idx_selected = idx_selected[0:1] # with shape 1
        else:
            idx_selected=torch.tensor([-100]).long()

        if user_id==0:
            idx=idx_selected.clone().detach()
        else:
            idx=torch.cat((idx,idx_selected.clone().detach()),dim=0)
    num_selected = torch.sum(idx > -100).cpu().data.numpy()
    return idx,num_selected

def apply_idx_batch(observed_train,pool_data,idx,start_idx):

    for idx_idx in range(idx.shape[0]):
        idx_sel=idx[idx_idx]
        if idx_sel!=-100:
            observed_train[start_idx+idx_idx,idx_sel]=pool_data[start_idx+idx_idx,idx_sel]
            pool_data[start_idx+idx_idx,idx_sel]=0.

        else:
            print('No data selected at row %s'%(start_idx+idx_idx))
    return observed_train,pool_data











def assign_zero_row_2D_with_target_reverse(data,data_target,value=0.):
    # This is the reverse operation s.t. the observed user will be assigned to 0.
    data=torch.tensor(data.data)
    sum_data=torch.sum(torch.abs(data_target),dim=1)
    data[sum_data>=0.001,:]=value
    return data # Can be seen as the entries of data matrix with the positions of the unobserved rows of the data_target are 0
def Random_select_idx(pool_data_cp,obs,step):
    flat_pool_data = pool_data_cp.view(1, -1)

    idx_obs = (torch.abs(flat_pool_data) > 0.).nonzero()
    if len(idx_obs) == 0:
        return [],0,True
    else:
        idx_obs = idx_obs[:, 1]
        if idx_obs.shape[0] <= step:
            idx_selected = idx_obs
            num_selected = idx_obs.shape[0]
        else:
            idx_random = torch.randperm(idx_obs.shape[0])
            idx_selected = idx_obs[idx_random]
            idx_selected = idx_selected[0:step]
            num_selected = step

        row = (idx_selected / obs).view(-1, 1)
        column = (idx_selected % obs).view(-1, 1)
        idx = torch.cat((row, column), dim=1)
        return idx,num_selected,False
def Random_select_idx_batch(pool_data_cp,obs,step,batch_size=100):
    # Mini batch version of random
    pool_data_cp_cp=pool_data_cp.clone().detach()
    # Check if possible to select
    total_active=torch.sum(torch.abs(pool_data_cp_cp)>0.)
    if total_active==0:
        return [], 0, True
    valid_idx=torch.unique((torch.abs(pool_data_cp_cp)>0.).nonzero()[:,0])
    if valid_idx.shape[0]<batch_size:
        batch_size=valid_idx.shape[0]
    rand_idx=torch.randperm(valid_idx.shape[0])[0:batch_size]
    valid_idx_batch=valid_idx[rand_idx]
    # check if active sample is enough for step
    pool_data_cp_batch = pool_data_cp_cp[valid_idx_batch, :]
    active_num = torch.sum(torch.abs(pool_data_cp_batch) > 0.)

    if active_num < step:
        new_step = active_num
    else:
        new_step = step

    # Assign rest to value
    valid_idx[rand_idx] = -1
    invalid_idx = valid_idx[valid_idx > -1]
    pool_data_cp_cp[invalid_idx, :] = 0

    # Start to pick
    flat_pool_data = pool_data_cp_cp.view(1, -1)

    idx_obs = (torch.abs(flat_pool_data) > 0.).nonzero()
    if len(idx_obs) == 0:
        return [], 0, True
    else:
        idx_obs = idx_obs[:, 1]
        if idx_obs.shape[0] <= new_step:
            idx_selected = idx_obs
            num_selected = idx_obs.shape[0]
        else:
            idx_random = torch.randperm(idx_obs.shape[0])
            idx_selected = idx_obs[idx_random]
            idx_selected = idx_selected[0:step]
            num_selected = step

        row = (idx_selected / obs).view(-1, 1)
        column = (idx_selected % obs).view(-1, 1)
        idx = torch.cat((row, column), dim=1)
        return idx, num_selected, False




def Compute_AUIC_Differece(source,**kwargs):
    # This is to compute the AUIC difference of source with other metric
    num_target=len(kwargs)
    Diff=[]

    for key,value in kwargs.items():
        Diff_comp=value-source
        Diff.append(Diff_comp)
    Diff=['%.3f'%(Diff_comp) for Diff_comp in Diff]
    return Diff
def Table_format(table,Results_list):
    table.add_row(Results_list)
    return table
def store_Table(filepath,table,title):
    with open(filepath,'w') as f:
        f.write(table.get_string(title=title))
def square_dict(D):
    D_squared=collections.OrderedDict()
    for key,value in D.items():
        D_squared[key]=value**2
    return D_squared
def W_sample_stat(W_sample):
    A=collections.defaultdict(dict)
    for key_W,value_W in W_sample.items():
        for key_layer,value_layer in value_W.items():

            A[key_layer][key_W]=value_layer
    W_mat_dict=collections.OrderedDict()
    for key_layer,value_layer in A.items():
        counter=0
        for key_sample,value_sample in value_layer.items():

            if counter==0:
                value_sample_comp=torch.unsqueeze(value_sample,dim=0)
                W_mat=value_sample_comp
            else:
                value_sample_comp = torch.unsqueeze(value_sample, dim=0)
                W_mat=torch.cat((W_mat,value_sample_comp),dim=0)
            counter+=1
        W_mat_dict[key_layer]=W_mat.cpu().data.numpy()
    W_mat_var=collections.OrderedDict()
    # Compute std
    for k,v in W_mat_dict.items():
        std_v=np.std(v,axis=0)
        W_mat_var[k]=std_v

    return W_mat_var

def zero_grad(list_p):
    for p in list_p:
        if p.grad is not None:
            p.grad.data.zero_()
def grap_modify_grad(list_p,W_dict_size,Data_N):
    grad_list=[]
    for p in list_p:
        if p.grad is not None:
            grad_list.append(-p.grad.clone().detach()/(W_dict_size*Data_N))
        else:
            grad_list.append('None')
    return grad_list
def assign_grad(list_p,list_grad):
    idx=0
    for p in list_p:
        if type(list_grad[idx])!=type('None'):
            p.grad.data=list_grad[idx].data.clone()
        else:
            p.grad.data=None
        idx+=1
def Combine_Results(**kwargs):
    tmp_dict = collections.defaultdict(dict)
    counter_part=1

    # Re-order the dict
    for key_dict,value_dict in kwargs.items():
        for key_results,value_results in value_dict.items():
            tmp_dict['%s'%(key_results)]['Part_%s'%(counter_part)]=value_results
        counter_part+=1
    # Combine the results
    results_dict = collections.defaultdict(dict)
    for key_results,value_results in tmp_dict.items():
        counter=0
        for key,value in value_results.items():
            if counter==0:
                Combined_Results=value
            else:
                Combined_Results=np.concatenate((Combined_Results,value),axis=0)
            counter+=1
        results_dict[key_results]=Combined_Results
    Results_obj=Results_Class(**dict(results_dict))
    return Results_obj

def reduce_size(samples,perm1,perm2,flag_one_row=False):
    # Note the samples have the shape N_w x N_z x N x obs or N_z x N x obs or N_w x Nz x Np x N x obs
    if flag_one_row==True:
        if len(samples.shape)==3:
            sample_reduce_1=samples[perm1,:,:]
            sample_reduce_2=sample_reduce_1[:,perm2,:]
        else:
            raise NotImplementedError
    else:
        if len(samples.shape)==4:
            sample_reduce_1=samples[perm1,:,:,:] # reduced x n_z x N x obs
            sample_reduce_2=sample_reduce_1[:,perm2,:,:]
        elif len(samples.shape)==3:
            sample_reduce_2=samples[perm2,:,:]
        elif len(samples.shape) == 5:

            sample_reduce_1 = samples[perm1,:, :, :, :]  # reduced x n_z x Np x N x obs
            sample_reduce_2 = sample_reduce_1[:, perm2, :,:, :] # reduce x reduce x Np x N x obs

    return sample_reduce_2


def selection_pattern(observed_data,observed_data_old):
    # This file is to detect the selection pattern (more scattered/focused for each user)
    old_data_mask=get_mask(observed_data_old)
    old_data_mask_column=(torch.sum(old_data_mask,dim=1,keepdim=True)>0.).float() # N x 1
    observed_data_wrt_old=observed_data*old_data_mask_column # N x obs
    observed_data_wrt_new=observed_data*(1-old_data_mask_column) # N x obs
    Diff_wrt_old=observed_data_wrt_old-observed_data_old
    if torch.sum(torch.abs(Diff_wrt_old))==0.:
        old_stat=0.
    else:
        Diff_wrt_old=remove_zero_row_2D(Diff_wrt_old)
        old_stat = torch.mean(torch.sum(torch.abs(Diff_wrt_old) > 0., dim=1).float())
    if torch.sum(torch.abs(observed_data_wrt_new))==0.:
        new_stat=0.
    else:
        Diff_wrt_new=remove_zero_row_2D(observed_data_wrt_new)
        new_stat=torch.mean(torch.sum(torch.abs(Diff_wrt_new)>0.,dim=1).float())
    return old_stat,new_stat
def new_selection_num(observed_data,observed_data_old):
    old_data_mask = get_mask(observed_data_old)
    old_data_mask_column = (torch.sum(old_data_mask, dim=1, keepdim=True) > 0.).float()  # N x 1
    observed_data_wrt_old = observed_data * old_data_mask_column  # N x obs
    observed_data_wrt_new = observed_data * (1 - old_data_mask_column)  # N x obs
    String=''
    if torch.sum(torch.abs(observed_data_wrt_new))==0.:
        String='None'
    else:
        Diff_wrt_new=remove_zero_row_2D(observed_data_wrt_new)
        new_stat=torch.sum(torch.abs(Diff_wrt_new)>0.,dim=1)
        unique_seq=torch.unique(new_stat)
        for num in unique_seq:
            select_num=torch.sum((new_stat==num))
            String=String+' %s:%s'%(num.cpu().data.numpy(),select_num.cpu().data.numpy())
    return String
def old_selection_num(observed_data,observed_data_old):
    old_data_mask = get_mask(observed_data_old)
    old_data_mask_column = (torch.sum(old_data_mask, dim=1, keepdim=True) > 0.).float()  # N x 1
    observed_data_wrt_old = observed_data * old_data_mask_column  # N x obs
    observed_data_wrt_new = observed_data * (1 - old_data_mask_column)  # N x obs
    # compute old difference
    observed_data_wrt_old=observed_data_wrt_old-observed_data_old
    String = ''
    if torch.sum(torch.abs(observed_data_wrt_old)) == 0.:
        String = 'None'

    else:
        Diff_wrt_old = remove_zero_row_2D(observed_data_wrt_old)
        old_stat = torch.sum(torch.abs(Diff_wrt_old) > 0., dim=1)
        unique_seq = torch.unique(old_stat)
        for num in unique_seq:
            select_num = torch.sum((old_stat == num))
            String = String + ' %s:%s' % (num.cpu().data.numpy(), select_num.cpu().data.numpy())
    return String

def outlier_detection(observed_data,observed_data_old,diff_scale=2.):
    # This file is to detect whether the new selected points has any outliers
    Diff_data=observed_data-observed_data_old
    Outlier_dim=[]
    flag_outlier=False
    outlier_value=[]
    for d in range(Diff_data.shape[-1]):
        column_diff=torch.unsqueeze(Diff_data[:,d],dim=1) # N x 1
        column_old=torch.unsqueeze(observed_data_old[:,d],dim=1) # N x 1
        if torch.sum(torch.abs(column_diff))>0.:
            column_diff_rz=remove_zero_row_2D(column_diff)
            column_old_rz=remove_zero_row_2D(column_old)
            column_diff_rz_exp=torch.unsqueeze(column_diff_rz,dim=1) # N x 1 x 1
            column_old_rz_exp=torch.unsqueeze(column_old_rz,dim=0).repeat(column_diff_rz.shape[0],1,1) # N_rz x N x 1
            Diff_column=torch.abs(column_diff_rz_exp-column_old_rz_exp)[:,:,0] # (N_rz x N )
            Diff_min=torch.max(torch.min(Diff_column,dim=1)[0])
            if Diff_min>diff_scale:
                Outlier_dim.append(d)
                flag_outlier=True
                outlier_diff=Diff_min.cpu().data.numpy()
                outlier_value.append('%.3f'%(outlier_diff.tolist()))

    if flag_outlier==True:
        return Outlier_dim,outlier_value
    else:
        return 'None','None'

def ReadPTable(filepath):
    counter=0
    with open(filepath) as f:
        for line in f:
            Result_comp = np.zeros((1,7))# selected,BALD,RAND,RANDP,DiffBALD,DiffRAND,DiffRandP
            selected_num_comp=re.search(r'[\d]+',line)
            data_comp=re.findall(r'[\d-]+\.\d+',line)
            if selected_num_comp is not None: # Data exists
                Result_comp[0,0]=float(selected_num_comp.group())
                Result_comp[0,1],Result_comp[0,2],Result_comp[0,3],Result_comp[0,4],Result_comp[0,5],Result_comp[0,6]=float(data_comp[0]),float(data_comp[1]),\
                                                             float(data_comp[2]),float(data_comp[3]),\
                               float(data_comp[4]),float(data_comp[5])
                if counter==0:
                    Results=Result_comp
                else:
                    Results=np.concatenate((Results,Result_comp),axis=0)
                counter+=1
    return Results

def Select_Movielens(data,size1=1000,size2=2000,base='feature'):
    # This is to select the features/users with most ratings, e.g. if base is feature, then it will select size1 of features with most ratings and
    # size2 of users with most ratings
    # data should be tensor
    if base=='feature':
        # number of selection
        choice=torch.sum(torch.abs(data) > 0., dim=0)  # obs_dim
        _,idx=torch.sort(choice,descending=True)
        croped_mat=data[:,idx[0:size1]]# N x size 1
        if croped_mat.shape[0]>size2:
            choice=torch.sum(torch.abs(croped_mat) > 0., dim=1) # N
            _,idx=torch.sort(choice,descending=True)
            crop_crop_mat=croped_mat[idx[0:size2],:] # size 2 x size 1
        else:
            crop_crop_mat=croped_mat
        #Shuffle the data
        shuffle_idx=torch.randperm(crop_crop_mat.shape[0])
        crop_crop_mat=crop_crop_mat[shuffle_idx,:]
        return crop_crop_mat
    else:
        raise NotImplementedError




def minibatch_BALD(BALD,step,batch_size=100,value=0):
    BALD=BALD.clone().detach()
    # Check if possible to select
    total_active=torch.sum(BALD>value)
    if total_active==0:
        # No available BALD block
        return BALD,0
    valid_idx=torch.unique((BALD>value).nonzero()[:,0])
    valid_size=valid_idx.shape[0]
    if valid_size<batch_size:
        batch_size=valid_size
    rand_idx=torch.randperm(valid_idx.shape[0])[0:batch_size]
    valid_idx_batch=valid_idx[rand_idx]

    # check if active sample is enough for step
    BALD_batch=BALD[valid_idx_batch,:]
    active_num=torch.sum(torch.abs(BALD_batch)>value)
    if active_num<step:
        new_step=active_num
    else:
        new_step=step
    # Assign rest to value
    valid_idx[rand_idx]=-1
    invalid_idx=valid_idx[valid_idx>-1]
    BALD[invalid_idx,:]=value

    return BALD,new_step



