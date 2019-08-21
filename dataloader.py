import torch
from torch.utils.data.dataset import Dataset
import os 
import numpy as np 
from sklearn.model_selection import train_test_split
class SmartWall(Dataset):
    def __init__(self, data,label):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = data.astype(np.single)
        self.label = label
        #self.data_raw = dict(np.load(data_path,allow_pickle=True))['arr_0']
        '''
        print(data_raw.shape)
        self.data=  np.expand_dims(data_raw[0]['data'],axis=0)
        self.label = np.expand_dims(data_raw[0]['label'],axis=0)
        for j in range(len(data_raw)): 
            try:
                self.data=np.concatenate((self.data,np.expand_dims(data_raw[j]['data'],axis=0)),axis=0)
                self.label=np.concatenate((self.label,np.expand_dims(data_raw[j]['label'],axis=0)),axis=0)
                if j % 100 ==0:
                    print('loding'+data_path+':',j)
            except ValueError :
                print('read_data\n',data_raw[j]['data'])               
        self.data = self.data
        '''
        #self.data_len=len(self.data_raw)

    def __getitem__(self, index):
        return torch.unsqueeze(torch.from_numpy(self.data[index,:,:]),0),self.label[index]
        '''
        temp = torch.unsqueeze(torch.cuda.FloatTensor(self.data_raw[index]['data']),-1)
        if temp.shape[0]==0:
            index -= 1
            while torch.unsqueeze(torch.cuda.FloatTensor(self.data_raw[index]['data']),-1).shape == 0:
                index -= 1
            single_data = torch.unsqueeze(torch.cuda.FloatTensor(self.data_raw[index]['data']),-1)
            single_label = self.data_raw[index]['label']
        else:
            single_data = temp
            single_label = self.data_raw[index]['label']           

        return (torch.transpose(single_data.repeat(4,1,1),0,2), single_label)
        '''
    def __len__(self):
        return self.data.shape[0]

def data_loader(train_batch_size,test_batch_size):
    x = np.load('/content/drive/My Drive/smart_wall_data/X.npy')
    print('x:',x.shape)
    y = np.load('/content/drive/My Drive/smart_wall_data/y.npy')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)

    train_dataset=SmartWall(x_train,y_train)
    test_dataset=SmartWall(x_test,y_test)
    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True)
    test_loader=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True
    )

    return train_loader,test_loader,len(test_dataset)