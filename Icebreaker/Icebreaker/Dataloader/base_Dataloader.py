import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import itertools
import zipfile
import requests
import numpy as np
import pickle
import json
from Util.Util_func import *
from sklearn.utils import shuffle
from sklearn import preprocessing
class base_UCI(object):
    '''Base class that stores the UCI data set'''
    def __init__(self,csv_file,root_dir,rs,test_size=0.2,flag_normalize=True,scheme='1',min_data=0.,normalization='True_Normal'):
        '''
        This is to load the UCI data as a whole
        :param csv_file: Which UCI files to load (range '0'-'12')
        :type csv_file: Str
        :param root_dir: The path where UCI data are stored
        :type root_dir: Str
        :param test_size: The percentage of test data in terms of the entire dataset
        :type test_size: Float
        :param flag_normalize: Whether to enable the pre-processing of the data
        :type flag_normalize: Bool
        :param rs: The random seed for splitting the training/test data
        :type rs: Int
        :param scheme: The train test scheme
        :type scheme: Str
        '''
        self.path = root_dir + '/d' + csv_file + '.xls'
        self.Data = pd.read_excel(self.path)
        self.Data_mat = self.Data.as_matrix()
        self.obs_dim = self.Data_mat.shape[1]
        self.test_size = test_size
        self.csv_file = csv_file
        self.flag_normalize=flag_normalize
        self.rs=rs
        # Normalize the data between 1 and 2
        if self.flag_normalize == True and self.csv_file != '12':
            min_Data = min_data
            max_Data = 1
            if normalization=='True_Normal':
                Data_std = preprocessing.scale(self.Data_mat)
                Data_std[Data_std == 0] = 0.01
                self.Data_mat = Data_std
            elif normalization=='0_1_Normal':
                Data_std = (self.Data_mat - self.Data_mat.min(axis=0)) / (
                        self.Data_mat.max(axis=0) - self.Data_mat.min(axis=0))
                self.Data_mat = Data_std + min_Data
                #self.Data_mat=self.Data_mat[0:180,:]
            else:
                raise NotImplementedError
            self.Data_mat=shuffle(self.Data_mat,random_state=self.rs)



        if scheme=='1':
            # Random mask value as test
            mask_train=base_generate_mask(self.Data_mat.shape[0],self.Data_mat.shape[1],mask_prop=self.test_size)
            mask_test=1-mask_train
            self.mask_train=mask_train
            self.mask_test=mask_test
            self.mask_other=mask_test
            self.mask_in=self.mask_train
            self.Data_train=self.Data_mat*self.mask_train.cpu().data.numpy()
            self.Data_test=self.Data_mat*self.mask_test.cpu().data.numpy()
            self.Data_other=self.Data_test
            self.Data_in=self.Data_train
        elif scheme=='2':
            # Random split row first as train data and test data, mask both dataset with test_size and use remaing ones in test dataset as input and missing ones as target

            # Train/Test split
            self.Data_train_orig, self.Data_test_orig, _, _ = train_test_split(self.Data_mat, self.Data_mat,
                                                                     test_size=0.1,
                                                                     random_state=rs)
            mask_train = base_generate_mask(self.Data_train_orig.shape[0], self.Data_train_orig.shape[1], mask_prop=self.test_size)
            mask_other=1-mask_train
            mask_in=base_generate_mask(self.Data_test_orig.shape[0], self.Data_test_orig.shape[1], mask_prop=self.test_size)
            mask_test=1-mask_in
            self.Data_train=self.Data_train_orig*mask_train.cpu().data.numpy()
            self.Data_other=self.Data_train_orig*mask_other.cpu().data.numpy()
            self.Data_in=self.Data_test_orig*mask_in.cpu().data.numpy()
            self.Data_test=self.Data_test_orig*mask_test.cpu().data.numpy()


        else:
            raise NotImplementedError





    def train_data(self):
        return self.Data_train
    def test_data(self):
        return self.Data_test



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


class ToTensor(object):
    '''
    Convert numpy array to Tensor
    '''
    def __call__(self,sample):
        return torch.from_numpy(sample).float()

class DeleteColumn(object):
    'Delete columns (This is used with streaming data and mapped movie index)'
    def __call__(self,sample,obs_dim):
        return sample[:,0:obs_dim]

# movielens base class (only support for 1M movielens)
class base_movielens(object):
    # Only used for download and process the movielens
    def __init__(self,file_path,download=True,flag_initial=True,pre_process_method='Fixed',flag_pickle=True,
                 user_ratio=0.1,test_ratio=0.25,rs=40
                 ):
        self.file_path=file_path
        self.flag_pickle=flag_pickle
        self.user_ratio=user_ratio
        self.test_ratio=test_ratio
        self.rs=rs

        if download==True:
            self.download()
            # Unzip File
            zip_ref=zipfile.ZipFile(file_path+'/ml-1m.zip','r')
            zip_ref.extractall(file_path)
            zip_ref.close()
        if flag_initial==True:
            # Read original data
            self.ratings_pd = pd.read_table(file_path + '/ml-1m/ratings.dat', sep='::',
                                            names=['user', 'movie', 'rating', 'time'])
            self.users_pd = pd.read_table(file_path + '/ml-1m/users.dat', sep='::',
                                          names=['user', 'gender', 'age', 'occupation', 'zip'])
            self.movie_pd = pd.read_table(file_path + '/ml-1m/movies.dat', sep='::', names=['movie', 'title', 'genre'])

            if pre_process_method=='Fixed':

                # matrix preprocess method
                train_batch,test_batch,overall_data=self._to_matrix()
                # Save the processed data
                self._save_data('Fixed_train_mat_user_%s'%(self.user_ratio),train_batch)
                self._save_data('Fixed_test_mat_user_%s'%(self.user_ratio),test_batch)
                self._save_data('Fixed_overall_mat',overall_data)
                self.train_batch=train_batch
                self.test_batch=test_batch
                self.overall_data=overall_data
                # Split useing test ratio
                mask_train, mask_train_other = base_generate_mask_incomplete(train_batch, mask_prop=self.test_ratio,
                                                                             rs=self.rs)
                # Generate actual train and test data
                mask_test_input, mask_test_tar = base_generate_mask_incomplete(test_batch, mask_prop=self.test_ratio, rs=self.rs)
                self.train_data=train_batch*mask_train
                self.test_data_input=self.test_batch*mask_test_input
                self.test_data_target=self.test_batch*mask_test_tar
                # Store Masks
                self.mask_train,self.mask_train_other,self.mask_test_input,self.mask_test_tar=mask_train,mask_train_other,mask_test_input,mask_test_tar
            elif pre_process_method=='Fixed_No_User':
                # This process method does not split test user and pure imputing training matrix.
                train_batch, test_batch, overall_data = self._to_matrix()
                self._save_data('Fixed_overall_mat', overall_data)
                self.overall_data=overall_data
                mask_train, mask_test = base_generate_mask_incomplete(self.overall_data, mask_prop=self.test_ratio,
                                                                             rs=self.rs)
                self.train_data=overall_data*mask_train
                self.test_data_input=self.train_data
                self.test_data_target=overall_data*mask_test
                self.mask_tfrain,self.mask_train_other,self.mask_test_input,self.mask_test_tar=mask_train,mask_test,mask_train,mask_test

            # Streaming preprocess method (also gives the matrix version data)
            elif pre_process_method=='Stream':
                raise NotImplementedError
                self.ratings,self.overall_data,self.train_matrix,self.test_matrix=self._to_stream(batch_user)
                self.num_row=batch_user
                self.num_column=self.ratings_pd['movie'].max()
                # Save training data
                self._save_ratings(file_path + '/ml-1m/'+self.file_name_stream,flag_pickle=self.flag_pickle) # Save the streaming data
                # Save completed data
                self._save_data(self.file_name_mat_overall, overall_data)
                # Save matrix train data
                self._save_data(self.file_name_train_mat,self.train_matrix)
                # Save matrix test data
                self._save_data(self.file_name_test_mat, self.test_matrix)
                #np.savetxt(file_path+'/ml-1m/'+self.file_name_mat,overall_data,delimiter=',')
        else:
            if pre_process_method=='Fixed':
            # Load stored data
                train_batch=self._load_data('Fixed_train_mat_user_%s.pkl'%(self.user_ratio))
                test_batch=self._load_data('Fixed_test_mat_user_%s.pkl'%(self.user_ratio))
                overall_data=self._load_data('Fixed_overall_mat.pkl')
                ###### Debug #########
                # test_batch[test_batch==1]=3

                ###### Debug ##########
                # train_batch_cp=copy.deepcopy(train_batch)
                # test_batch_cp=copy.deepcopy(test_batch)
                # overall_data_cp=copy.deepcopy(overall_data)
                # train_batch[train_batch_cp==1]=0
                # test_batch[test_batch_cp==1]=0
                # overall_data[overall_data_cp==1]=0
                #
                # train_batch[train_batch_cp == 5] = 1
                # test_batch[test_batch_cp == 5] = 1
                # overall_data[overall_data_cp == 5] = 1
                # train_batch[train_batch_cp == 4] = 1
                # test_batch[test_batch_cp == 4] = 1
                # overall_data[overall_data_cp == 4] = 1

                ####### Debug #########
                # train_batch[train_batch==1]=0
                # test_batch[test_batch==1]=0
                # overall_data[overall_data==1]=0
                ####### Debug #########

                # train_batch=train_batch-1
                # test_batch=test_batch-1
                # overall_data=overall_data-1

                ######## DEBUG Purpose ############

                # d_load = np.load('/home/glovev659248/Projects/SEDDI/Dataloader/mlen1m.npz')
                # Data=d_load['Data']
                # train_batch,test_batch,_,_=train_test_split(Data,Data,test_size=0.1,random_state=self.rs,shuffle=True)
                # overall_data=Data
                ###################################




                self.train_batch,self.test_batch,self.overall_data=train_batch,test_batch,overall_data
                mask_train, mask_train_other = base_generate_mask_incomplete(train_batch, mask_prop=self.test_ratio,
                                                                         rs=self.rs)
                # Generate actual train and test data
                mask_test_input, mask_test_tar = base_generate_mask_incomplete(test_batch, mask_prop=self.test_ratio,
                                                                           rs=self.rs)
                self.train_data = train_batch * mask_train
                self.train_other=train_batch*mask_train_other
                self.test_data_input = self.test_batch * mask_test_input
                self.test_data_target = self.test_batch * mask_test_tar
                # Store Masks
                self.mask_train, self.mask_train_other, self.mask_test_input, self.mask_test_tar = mask_train, mask_train_other, mask_test_input, mask_test_tar
            elif pre_process_method=='Fixed_No_User':
                overall_data = self._load_data('Fixed_overall_mat.pkl')
                self.overall_data = overall_data
                mask_train, mask_test = base_generate_mask_incomplete(self.overall_data, mask_prop=self.test_ratio,
                                                                      rs=self.rs)
                self.train_data = overall_data * mask_train
                self.test_data_input = self.train_data
                self.test_data_target = overall_data * mask_test
                self.train_other=overall_data*mask_test
                self.mask_train, self.mask_train_other, self.mask_test_input, self.mask_test_tar = mask_train, mask_test, mask_train, mask_test
            elif pre_process_method=='Stream':
                raise NotImplementedError
                ''' Read processed data'''
                self.ratings_pd = pd.read_table(file_path + '/ml-1m/ratings.dat', sep='::',
                                                names=['user', 'movie', 'rating', 'time'])
                self.users_pd = pd.read_table(file_path + '/ml-1m/users.dat', sep='::',
                                              names=['user', 'gender', 'age', 'occupation', 'zip'])
                self.movie_pd = pd.read_table(file_path + '/ml-1m/movies.dat', sep='::', names=['movie', 'title', 'genre'])
                if self.flag_pickle==False:
                    self.ratings=np.loadtxt(file_path+'/ml-1m/'+load_file)
                    self.num_row=self.ratings.shape[0]
                    self.num_column=self.ratings.shape[1]
                else:
                    pkl_file=open(file_path+'/ml-1m/'+load_file,'rb')
                    self.ratings=pickle.load(pkl_file)
                    pkl_file.close()

                    # load train and test matrix
                    self.train_matrix=self._load_data('Stream_train_matdata_user_%s_test_%s.pkl'%(batch_user,test_size))
                    self.test_matrix = self._load_data(
                        'Stream_test_matdata_user_%s_test_%s.pkl' % (batch_user, test_size))
                    self.overall_data=self._load_data('Stream_overall_matdata_user_100_test_0.1.pkl')
            # Train/Test Split
            # self.ratings_train,self.ratings_test,_,_=train_test_split(self.ratings, self.ratings, test_size=test_size,
            #                  random_state=None,shuffle=False)
    def _to_matrix(self):
        # Extract all the data
        counter_row=0
        overall_data= np.zeros((self.ratings_pd['user'].max(), self.ratings_pd['movie'].max()))
        for idx, row in self.ratings_pd.iterrows():
            counter_row+=1
            user_idx,movie_idx,rating=row['user'],row['movie'],row['rating']
            overall_data[user_idx - 1, movie_idx - 1]=rating
            if counter_row%10000==0:
                print('Processed Row:%s'%(counter_row))
        # Split acoording to user ratio
        train_batch,test_batch,_,_=train_test_split(overall_data,overall_data,test_size=self.user_ratio,random_state=self.rs,shuffle=True)
        # Maksing train/test ratio values
        #mask_train,mask_train_other=base_generate_mask_incomplete(train_batch, mask_prop=self.test_ratio, rs=self.rs)
        #mask_test_input, mask_test_obs = base_generate_mask_incomplete(test_batch, mask_prop=self.test_ratio, rs=self.rs)
        return train_batch,test_batch,overall_data
    def _save_ratings(self,filename,flag_pickle=True):
        if flag_pickle==False:
            np.savetxt(filename,self.ratings,delimiter=',')
        else:
            savefile=open(filename+'.pkl','wb')
            pickle.dump(self.ratings,savefile)
            savefile.close()
    def _save_data(self,filename,data):
        savefile = open(self.file_path + '/ml-1m/' + filename + '.pkl', 'wb')
        pickle.dump(data, savefile)
        savefile.close()
    def _load_data(self,filename):
        pkl_file = open(self.file_path + '/ml-1m/' + filename, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data
    def _to_stream(self,user_batch=100):
        rating_sort_time=self.ratings_pd.sort_values(by=['time'])
        rating_sort_time=rating_sort_time.reset_index(drop=True)
        row_idx_history=0
        seen_row=0
        counter=0 # count time

        # This is used to store complete and test matrix
        overall_data = np.zeros((self.ratings_pd['user'].max(), self.ratings_pd['movie'].max()))
        test_matrix=np.zeros((self.ratings_pd['user'].max(),self.ratings_pd['movie'].max())) # num_user x num_movie (based on movie id)

        # This is used to store the history
        train_matrix = np.zeros((self.ratings_pd['user'].max(),
                                 self.ratings_pd['movie'].max()))  # num_user x num_movie (based on movie id )

        overall_user_list=[]
        while True:
            # Go through the entire table until the end
            stream_matrix=np.zeros((user_batch,self.ratings_pd['movie'].max()))

            observed_user_list=[]
            add_user_count=0
            print('Seen row:%s'%(seen_row))

            for idx,row in rating_sort_time.iterrows():
                # Go throught the row of the table
                drop=np.random.rand(1)<=self.test_size # Indicate the current row is masked for test purpose

                user_idx,movie_idx,rating=row['user'],row['movie'],row['rating']
                # Applied movie_idx mapping
                movie_idx=self.idx_mapping['%s'%(movie_idx)]

                if user_idx not in observed_user_list:
                    # Check history
                    if user_idx in overall_user_list:
                        # restore history
                        history=train_matrix[user_idx-1,:]
                        stream_matrix[add_user_count,:]=history
                    else:
                        overall_user_list.append(user_idx)


                    observed_user_list.append(user_idx)
                    if drop==False:
                        stream_matrix[add_user_count,movie_idx-1]=rating
                        overall_data[user_idx-1,movie_idx-1]=rating
                        train_matrix[user_idx-1,movie_idx-1]=rating
                        test_matrix[user_idx-1,movie_idx-1]=0
                    else:
                        stream_matrix[add_user_count, movie_idx - 1] = 0
                        overall_data[user_idx - 1, movie_idx - 1] = rating
                        train_matrix[user_idx - 1, movie_idx - 1] = 0
                        test_matrix[user_idx-1,movie_idx-1]=rating
                    add_user_count+=1
                else:
                    # restore history
                    history = train_matrix[user_idx-1, :]


                    idx_user_in_batch=observed_user_list.index(user_idx)
                    stream_matrix[idx_user_in_batch, :] = history
                    if drop==False:
                        stream_matrix[idx_user_in_batch,movie_idx-1]=rating
                        overall_data[user_idx-1, movie_idx-1] = rating
                        train_matrix[user_idx - 1, movie_idx - 1]=rating
                        test_matrix[user_idx-1,movie_idx-1]=0
                    else:
                        stream_matrix[idx_user_in_batch, movie_idx - 1] = 0
                        overall_data[user_idx - 1, movie_idx - 1] = rating
                        train_matrix[user_idx - 1, movie_idx - 1] = 0
                        test_matrix[user_idx - 1, movie_idx - 1] = rating

                if add_user_count==user_batch:
                    #check next user_idx if the same as the last one, complete this before stop
                    next_user_id=rating_sort_time.iloc[idx+1]['user']
                    if next_user_id!=user_idx:
                        # Current time finished

                        break
            #Throw away prevous seen data
            row_idx_history=idx
            seen_row+=idx+1
            rating_sort_time=rating_sort_time.drop(rating_sort_time.index[0:row_idx_history+1])
            rating_sort_time = rating_sort_time.reset_index(drop=True)
            if counter==0:
                Stream_Data=np.expand_dims(stream_matrix,axis=0)
            else:
                Stream_Data=np.concatenate((Stream_Data,np.expand_dims(stream_matrix,axis=0)),axis=0)
            counter+=1
            if seen_row==self.ratings_pd.shape[0]:
                break
        return Stream_Data,overall_data,train_matrix,test_matrix





    def download(self):
        'Download the Movielens'
        url='http://files.grouplens.org/datasets/movielens/ml-1m.zip'

        req=requests.get(url,stream=True)
        req.raise_for_status()
        with open(self.file_path+'/ml-1m.zip','wb') as fd:
            for chunk in req.iter_content(chunk_size=2**20):
                fd.write(chunk)


def create_movie_idx_mapping(File_rating,Filename_save,flag_original=False):
    # Read the rating file
    ratings_pd = pd.read_table(File_rating, sep='::',
                                    names=['user', 'movie', 'rating', 'time'])
    rating_sort_time = ratings_pd.sort_values(by=['time'])
    rating_sort_time = rating_sort_time.reset_index(drop=True)
    movie_idx_original=rating_sort_time['movie'].as_matrix()
    idx_map={}
    if flag_original==True:
        unique_idx=pd.unique(movie_idx_original)
        for i in unique_idx:
            idx_map['%s'%(i)]=int(i)

    else:
        unique_idx = pd.unique(movie_idx_original)
        for new_idx,old_idx in enumerate(unique_idx):
            idx_map['%s'%(old_idx)]=int(new_idx+1)
    with open(Filename_save,'w') as fp:
        json.dump(idx_map,fp)
    return None

def load_idx_mapping(Filename):
    with open(Filename,'r') as fp:
        idx_map=json.load(fp)
    return idx_map
def idx_zero_column(batch_ratings):
    '''ONLY APPLICABLE For The INITIAL TIME BATCH, NOT USE THIS FOR LATER TIME BATCH'''
    zero_idx = np.where(np.sum(batch_ratings, axis=0) == 0)[0]
    group_zero_idx = consecutive(zero_idx, stepsize=1)
    obs_dim = group_zero_idx[-1][0]
    return obs_dim
def num_new_movies(old_movie_num,new_batch,flag_torch=False):
    # check if already observed all movies
    if old_movie_num<new_batch.shape[1]:
        deleted_batch_ratings=new_batch[0:,old_movie_num:]
        if flag_torch==False:
            #num_new=np.min(np.where(np.sum(deleted_batch_ratings, axis=0) == 0)[0])
            zero_idx=np.where(np.sum(deleted_batch_ratings, axis=0) == 0)[0]
            group_zero_idx=consecutive(zero_idx,stepsize=1)
            num_new=group_zero_idx[-1][0]
        else:
            raise NotImplementedError
            num_new=torch.min((torch.sum(deleted_batch_ratings,dim=0)==0).nonzero())
    else:
        num_new=0
    return num_new
def delete_column(batch_data,obs_dim):
    return batch_data[:,0:obs_dim]
class movielens_dataset_batch(Dataset):
    def __init__(self,batch_data,transform=None,flag_GPU=True):
        self.batch_data=batch_data
        self.transform=transform
        self.flag_GPU=flag_GPU
    def __len__(self):
        return self.batch_data.shape[0]
    def __getitem__(self, idx):
        sample=self.batch_data[idx,:]
        if self.transform:
            sample=self.transform(sample)
            if self.flag_GPU==True:
                sample=sample.cuda()
        return sample
def restore_orig_matrix(current_matrix,dict_idx):
    orig_idx_list=[]
    current_idx_list=[]
    for key, value in dict_idx.items():
        orig_idx_list.append(int(key)-1)
        current_idx_list.append(value-1)
    # to numpy
    orig_idx_list=np.array(orig_idx_list)
    current_idx_list=np.array(current_idx_list)
    # Sort
    sort_index=np.argsort(current_idx_list)
    orig_idx_list=orig_idx_list[sort_index]
    return current_matrix[:,orig_idx_list]

def delete_zero_row(train_data):
    sum_data=torch.sum(torch.abs(train_data),dim=1) # N
    deleted_train_data=train_data[sum_data>0,:]
    return torch.tensor(deleted_train_data.data)
