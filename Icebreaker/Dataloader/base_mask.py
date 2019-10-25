import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

def base_generate_mask(batch_size,obs_dim,mask_prop=0.7):
    mask=torch.rand(batch_size,obs_dim)>mask_prop
    # rand index is set to 1
    # rand_index=torch.randint(low=0,high=obs_dim,size=(batch_size,))
    # mask[range(batch_size),rand_index.long()]=1
    return mask.float()

def get_mask(deleted_column_batch_data,flag_tensor=True):
    if flag_tensor==True:
        return (torch.abs(deleted_column_batch_data)>0.).float()
    else:
        return (np.abs(deleted_column_batch_data)>0.).astype(float)
def merge_mask(mask1,mask2,flag_tensor=True):
    assert mask1.shape==mask2.shape,'Masks with inconsistent shape'
    mask1[mask1!=mask2]=1.
    if flag_tensor:
        return mask1.float()
    else:
        return mask1.astype(float)
def base_generate_mask_incomplete(batch_data,mask_prop=0.25,rs=40):
    mask=np.zeros((batch_data.shape[0],batch_data.shape[1]))
    mask_other=np.zeros((batch_data.shape[0],batch_data.shape[1]))
    for row_idx in range(batch_data.shape[0]):
        pos = np.where(np.abs(batch_data[row_idx,:])>0)[0] # Positions of all observed data
        pos_train, pos_test, pos_train, pos_test = train_test_split(pos, pos, test_size=mask_prop) # select which is masked for train
        mask[row_idx, pos_train] = 1
        mask_other[row_idx, pos_test] = 1
    return mask,mask_other
def generate_drop_mask_rest(X,drop_prop=0.5):
    mask_current=get_mask(X) # Tensor with N x obs
    rand_matrix=torch.rand(X.shape) # Random tensor matrix with shape N x obs
    rand_matrix=rand_matrix*mask_current # Left masked random matrix
    rand_matrix_mask=(rand_matrix>0).float()
    mask_second=(rand_matrix>drop_prop).float() # Drop the masked values with shape N x obs
    mask_other=rand_matrix_mask-mask_second # Which is droped
    return mask_second,mask_other

def least_one_left_mask(mask,mask_drop):
    mask_dropped=mask*mask_drop
    zero_row=torch.sum(torch.abs(mask_dropped),dim=1)==0
    mask_np=mask.cpu().data.numpy()
    mask_one=np.zeros((mask_np.shape[0],mask_np.shape[1]))
    for row_idx in range(mask_np.shape[0]):
        pos= np.where(np.abs(mask_np[row_idx,:])>0)[0]
        if len(pos)>1:
            pos_train, pos_test, pos_train, pos_test = train_test_split(pos, pos, test_size=1)
        else:
            pos_test=pos
        mask_one[row_idx, pos_test] = 1
    mask_one=torch.from_numpy(mask_one).float().cuda()

    mask_dropped[zero_row,:]=mask_one[zero_row,:]
    return mask_dropped