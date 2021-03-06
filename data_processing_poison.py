import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import missingno
import numpy as np
import matplotlib.pyplot as plt
import poison_config

class data_processing_poison:
    def __init__(self):
        pass

    def normalize(self,data):
        '''
        Normalizes an array
        (subtract mean and divide by standard deviation)
        '''
        eps = 0.001
        # print(np.shape(data))
        if np.std(data) != 0:
            data = (data - np.mean(data)) / np.std(data)
        else:
            data = (data - np.mean(data)) / eps
        return data

    def normalize_dataset(self,x):
        '''
        Normalizes list of arrays
        (subtract mean and divide by standard deviation)
        '''
        normalized_dataset = []
        for data in x:
            normalized = self.normalize(data)
            normalized_dataset.append(normalized)
        return normalized_dataset
            
    def make_timesteps(self,x_data, y_data, num_steps):
      X = []
      y = []
      #print("In Time Steps\nAppend num_steps:{} rows\n".format(num_steps))
      #Time steps drops length of arr - num_steps to length of arr - so add num_steps at end to get all
      x_data_plus_steps = np.copy(x_data)
      y_data_plus_steps = np.copy(y_data)
      for x in range(num_steps):
        x_data_plus_steps = np.vstack([x_data_plus_steps, np.array(x_data[x])])
      for j in range(num_steps):
        y_data_plus_steps = np.vstack([y_data_plus_steps,np.array(y_data[j])])

      #print("\nCheck Length of x/y_data: {}|{} vs x/y_data_plus_steps: {}|{}\n:::: Start Loop ::::\n".format(x_data.shape[0],y_data.shape[0],x_data_plus_steps.shape[0],y_data_plus_steps.shape[0]))
      #print("x_data_plus\n{}".format(x_data_plus_steps))
      #print("y_data_plus\n{}".format(y_data_plus_steps))
      #use the length of original array for iterations
      for i in range(x_data.shape[0]):
        #new sliding window index
        end_ix = i + num_steps
        seq_X = x_data_plus_steps[i:end_ix]
        seq_y = y_data_plus_steps[end_ix]
        seq_y = float(seq_y)
        X.append(seq_X)
        y.append(seq_y)
        #print("i:{} | end_ix:{} |\nseq_X:\n{}|\nseq_y:\n{}".format(i,end_ix,seq_X,seq_y))

      #print("Make output arrs:\nLen of X:{}\n".format(len(X)))
      x_array = np.array(X)
      y_array = np.vstack([np.array(i) for i in y])

      #print("Check outputs:\nx_array shape : {}\n{}\n\ny_array shape : {}\n{}\n".format(x_array.shape,x_array,y_array.shape,y_array))
      return x_array, y_array

    
    # First call to load the two csv files.  one file is normal traffic,
    # other file is attack or abnormal traffic.  Timesteps are used as 
    # an LSTM parameter to help gain contextual knowledge of the different
    # data points. 
    def load_data(self,path,file_path_normal,timesteps=80):
        # Create a normal dictiionary
        df = pd.read_csv(file_path_normal)
        print(list(df.columns))
        print()
        attack_distribution = df.groupby(by='Y').size()
        with open(path + 'poison_tng_log.txt','a') as f:
            f.write("\nData Distributions:\n")
            f.write(str(attack_distribution))
        f.close()
        print("normal (0) abnormal(1)", attack_distribution)
        print()
        
        #SHuffle been running at fasle???? change it back?
        train, test = train_test_split(df, test_size=0.25, random_state=0, shuffle=False)
        train = np.array(train)
        test = np.array(test)
        x_train = np.delete(train,poison_config.POISON_FEATURES,1)
        y_train_asf_rm = np.delete(train,0,1)
        y_train_fg_rm = np.delete(y_train_asf_rm,0,1)
        y_train_mn_rm = np.delete(y_train_fg_rm,0,1)
        y_train_ed_rm = np.delete(y_train_mn_rm,0,1)
        y_train_lg_rm = np.delete(y_train_ed_rm,0,1)
        #y_train_jc_rm = np.delete(y_train_lg_rm,0,1)
        y_train_ndT_rm = np.delete(y_train_lg_rm,0,1)
        #summ  = np.delete(y_train_ndT_rm,0,1)
        #sum1  = np.delete(summ,0,1)
        #sum5 = np.delete(sum1,0,1)
        #sum10 = np.delete(sum5,0,1)
        #dif1 = np.delete(sum10,0,1)
        #avg5 = np.delete(dif1,0,1)
        #avg10 = np.delete(avg5,0,1)
        #std5 = np.delete(avg10,0,1)
        #std10 = np.delete(std5,0,1)
        y_train = np.delete(y_train_ndT_rm,0,1)
        x_test = np.delete(test,poison_config.POISON_FEATURES,1)
        y_test_asf_rm = np.delete(test,0,1)
        y_test_fg_rm = np.delete(y_test_asf_rm,0,1)
        y_test_mn_rm = np.delete(y_test_fg_rm,0,1)
        y_test_ed_rm = np.delete(y_test_mn_rm,0,1)
        y_test_lg_rm = np.delete(y_test_ed_rm,0,1)
        #y_test_jc_rm = np.delete(y_test_lg_rm,0,1)
        y_test_ndT_rm = np.delete(y_test_lg_rm,0,1)
        #y_summ  = np.delete(y_test_ndT_rm,0,1)
        #y_sum1  = np.delete(y_summ,0,1)
        #y_sum5 = np.delete(y_sum1,0,1)
        #y_sum10 = np.delete(y_sum5,0,1)
        #y_dif1 = np.delete(y_sum10,0,1)
        #y_avg5 = np.delete(y_dif1,0,1)
        #y_avg10 = np.delete(y_avg5,0,1)
        #y_std5 = np.delete(y_avg10,0,1)
        #y_std10 = np.delete(y_std5,0,1)
        y_test = np.delete(y_test_ndT_rm,0,1)

        #Create two classes and use keras.utils to create binary label arrays
        #CLASSES = ['normal', 'abnormal']
        ## this is one hot encoding for binary classificaiton
        ##y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
        #y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
        
        #y_train = np.asarray(y_train)
        #y_test = np.asarray(y_test)

        # Use make timesteps for LSTM timesteps.
        #print("sending x_train and y_train to make times steps {}".format(poison_timesteps))
        x_train,y_train= self.make_timesteps(np.array(x_train),np.array(y_train),poison_config.POISON_TIMESTEPS)
        #print("sending x_test and y_test to make times steps {}".format(poison_timesteps))
        x_test, y_test = self.make_timesteps(np.array(x_test), np.array(y_test), poison_config.POISON_TIMESTEPS)  

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        #print("x_train shape after reshape {}".format(x_train.shape))
        #print("x_test shape  after reshape {}".format(x_test.shape))

        return x_train, x_test, y_train, y_test