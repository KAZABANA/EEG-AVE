# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:54:21 2021

@author: user
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import scipy.io as scio
from torch.optim import Adam
from typing import Optional
from model_mutiscale import DE_mutiscale_classifier_esemble_twoclass,LabelSmoothingCrossEntropy
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
from typing import List, Dict
from Adversarial import DomainAdversarialLoss
class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.relu1(self.layer1(x))
        output = self.sigmoid(self.layer3(x))
        return output

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr_mult": 1}]

class StepwiseLR_GRL:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75,max_iter: Optional[float] = 100):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter
    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)##对参数进行xavier初始化，为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
        m.bias.data.zero_()
def get_weight(model):
    weight_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            weight_list.append(weight)
    return weight_list
def regularization_loss(weight_list, weight_decay, p=2):
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
    L2_list=['conv1.weight',
             'conv2.weight']
    reg_loss=0
    for name, w in weight_list:
        if name in L2_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
    reg_loss=weight_decay*reg_loss
    return reg_loss

def compute_acc(label_p,label_g):
    possible=label_p.cpu().detach().numpy()
    label_g=label_g.cpu().detach().numpy() 
    tru=0
    for i in range(0,len(label_g)):
        index_labels=np.argmax(label_g[i])
        index_possible=np.argmax(possible[i])
        if index_labels==index_possible:
            tru=tru+1
    return tru/len(label_g)

def test_model(loader_valid,classifier):
    right_num=0
    sample_num=0
    with torch.no_grad():
        classifier.eval()
        for step,(batch_x,batch_y) in enumerate(loader_valid):

            # wrap them in Variable
            inputs,labels =Variable(batch_x.cuda(0)), Variable(batch_y.cuda(0))
            predict,_=classifier(inputs)
            predict_label=torch.nn.functional.softmax(predict, dim=1)
            predict_label=predict_label.cpu().detach().numpy()
            labels=labels.cpu().detach().numpy()
            for i in range(0,len(labels)):
                if np.argmax(predict_label[i])==labels[i]:
                    right_num+=1
            sample_num+=len(labels)      
    acc=right_num/sample_num
    return acc

def test_model_esemble_smooth(loader_valid,classifier1,classifier2,classifier3):## esemble prediction function
    truth=[]
    prediction=[]
    prediction_1=[]
    prediction_2=[]
    prediction_3=[]
    with torch.no_grad():
        classifier1.eval()
        classifier2.eval()
        classifier3.eval()
        for step,(batch_x_1,batch_x_2,batch_x_3,batch_y) in enumerate(loader_valid):

            # wrap them in Variable
            inputs_1,inputs_2,inputs_3,labels =Variable(batch_x_1.cuda(0)),Variable(batch_x_2.cuda(0)),Variable(batch_x_3.cuda(0)),Variable(batch_y.cuda(0))
            predict_1,_=classifier1(inputs_1)
            predict_2,_=classifier2(inputs_2)
            predict_3,_=classifier3(inputs_3)
            predict_label_1=torch.nn.functional.softmax(predict_1, dim=1)
            predict_label_1=predict_label_1.cpu().detach().numpy()
            predict_label_2=torch.nn.functional.softmax(predict_2, dim=1)
            predict_label_2=predict_label_2.cpu().detach().numpy()
            predict_label_3=torch.nn.functional.softmax(predict_3, dim=1)
            predict_label_3=predict_label_3.cpu().detach().numpy()
            labels=labels.cpu().detach().numpy()
            for i in range(0,len(labels)):
                if np.argmax(predict_label_1[i])+np.argmax(predict_label_2[i])+np.argmax(predict_label_3[i])>1.5:
                    predict=1
                else:
                    predict=0
                prediction.append(predict)
                prediction_1.append(np.argmax(predict_label_1[i]))
                prediction_2.append(np.argmax(predict_label_2[i]))
                prediction_3.append(np.argmax(predict_label_3[i]))
                truth.append(labels[i])  
    truth=np.reshape(truth,(2400,1))
    predict_mat=[]
    predict_mat.append(np.reshape(prediction,(2400,1)))
    predict_mat.append(np.reshape(prediction_1,(2400,1)))
    predict_mat.append(np.reshape(prediction_2,(2400,1)))
    predict_mat.append(np.reshape(prediction_3,(2400,1)))
    predict_mat=np.hstack(predict_mat)
    predict_mat_smooth=np.copy(predict_mat)
    for j in range(4):
        for i in range(40):
            if np.mean(predict_mat_smooth[i*60:(i+1)*60,j])>=0.5: ## smoothing the prediction squence belong to the same trial
                predict_mat_smooth[i*60:(i+1)*60,j]=1
            else:
                predict_mat_smooth[i*60:(i+1)*60,j]=0
    acc=np.sum(truth.T==predict_mat_smooth[:,0])/2400
    return acc,predict_mat,predict_mat_smooth,truth

def max_min_scale(data_sque):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    data_sque=min_max_scaler.fit_transform(data_sque)
    return data_sque

def train_model(loader_train, loader_test,classifier, dann_loss, optimizer,scheduler):
    ## if you want to train a model with dann just set weight=1(Msdann), otherwise set weight=0 (Msnn)
    # switch to train mode
    weight=1
    classifier.train()
    dann_loss.train()
    crition= LabelSmoothingCrossEntropy()
    train_source_iter, train_target_iter=enumerate(loader_train),enumerate(loader_test)
    # train_source_iter and train_target_iter is data iterator that will never stop producing data
    T =33
#    scheduler.step()
    for i in range(T):
        _,(x_s,labels_s) = next(train_source_iter)
        x_s,labels_s=Variable(x_s.cuda(0)), Variable(labels_s.cuda(0)) 
        # data from target domain
        _,(x_t,_) = next(train_target_iter)
        x_t=Variable(x_t.cuda(0))
        input_st=torch.cat((x_s,x_t),dim=0)
        # compute output
        y_st,f_shallow= classifier(input_st)

        # cross entropy loss on source domain
        cls_loss = crition(y_st[0:48,:],labels_s.reshape(len(labels_s),1))
        # domain adversarial loss( the gaussian noise will benefit the performance of dann)
        transfer_loss = dann_loss(f_shallow[0:48,:]+0.005*torch.randn((48,64)).cuda(0),f_shallow[48:96,:]+0.005*torch.randn((48,64)).cuda(0))
#        transfer_loss = dann_loss(f_shallow[0:48,:].cuda(0),f_shallow[48:96,:].cuda(0))
        loss = cls_loss+weight*transfer_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('transfer_loss:',str( transfer_loss.data))
    print('cls_loss:',str(cls_loss.data))


def train_and_test_GAN(test_id,resolution,emotion_dim,max_iter,smooth_flag):## train a single resolution classifer based on dann
     
    BATCH_SIZE = 48
    hci_data_train,hci_data_test,hci_label_train,hci_label_test=get_dataset(resolution,test_id,emotion_dim,smooth_flag)
    torch_dataset_train = Data.TensorDataset(hci_data_train,hci_label_train)
    torch_dataset_test=Data.TensorDataset(hci_data_test,hci_label_test)
    loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
            )
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
            )
    classifier=DE_mutiscale_classifier_esemble_twoclass(32,resolution).cuda()
    classifier.apply(weigth_init)
    domain_discriminator = DomainDiscriminator(in_feature=64, hidden_size=128).cuda()
    dann_loss = DomainAdversarialLoss(domain_discriminator).cuda()
    optimizer = Adam(classifier.get_parameters() + domain_discriminator.get_parameters(),
                lr=0.0001, weight_decay=1e-3)
    lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=0.0001, gamma=10, decay_rate=0.75,max_iter=4000)
    best_acc1 = 0.
    if resolution==173:
        save_path='F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point25_deap'
    if resolution==91:
        save_path='F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point5_deap'
    if resolution==50:
        save_path='F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_1_deap'
    for epoch in range(max_iter):
        # train for one epoch
        train_model(loader_train,loader_test,classifier,dann_loss,optimizer,lr_scheduler)
    # evaluate on validation set
        acc1 =  test_model(loader_test,classifier)
        print('epoch:',str(epoch))
        print('test_acc:',str(acc1))
        #     # remember best acc@1
        if acc1>best_acc1:
            os.chdir(save_path)
            if emotion_dim==1:
                torch.save(classifier.state_dict(),'bestclassifier_arousal_'+str(test_id)+'_resolution_0_25'+'.pkl')
            elif emotion_dim==0:
                torch.save(classifier.state_dict(),'bestclassifier_'+str(test_id)+'_resolution_0_25'+'.pkl')
            else:
                torch.save(classifier.state_dict(),'bestclassifier_dominance_'+str(test_id)+'.pkl')
        best_acc1 = max(acc1, best_acc1)
    return best_acc1

def get_dataset(resolution,test_id,emotion_dim,smooth_flag):
    ## loading de feature under different resolution
    ## emotion_dim=1: Valence, emotion_dim=0: Arousal
    os.chdir('F:\\zhourushuang\\Multi-Scale\\data_for_learing')
    if resolution==173:
        deap_info_part1='total_band_part2_no_baseline.mat'
        deap_info_part2='total_band_part2_no_baseline.mat'
    if resolution==91:
        deap_info_part1='total_band_part2_no_baseline-0.5.mat'
        deap_info_part2='total_band_part2_no_baseline-0.5.mat'
    if resolution==50:
        deap_info_part1='total_band_part1_no_baseline-1.mat'
        deap_info_part2='total_band_part2_no_baseline-1.mat'
    deap_data_1 = scio.loadmat(deap_info_part1)['feature_struct'][0,0]
    deap_data_2 = scio.loadmat(deap_info_part2)['feature_struct'][0,0]
    deap_label = scio.loadmat(deap_info_part1)['feature_struct'][0,1]
    deap_data = np.row_stack((deap_data_1,deap_data_2))
    if smooth_flag==1:
        deap_data=smooth(deap_data,'hull',resolution)
    deap_data=np.reshape(deap_data,(np.shape(deap_data)[0],1,32,np.shape(deap_data)[2]))
    deap_label=deap_label.astype('int')
    deap_data=deap_data.astype('float32')
    deap_label[np.where(deap_label<=5)]=0
    deap_label[np.where(deap_label>5)]=1
    deap_label=deap_label[:,emotion_dim]
    deap_label=deap_label.reshape(len(deap_label),1)
    deap_data_test=data_norm(deap_data[2400*test_id:2400*(test_id+1),:,:],resolution)
    deap_label_test=deap_label[2400*test_id:2400*(test_id+1)]
    deap_label=np.delete(deap_label,np.arange(2400*test_id,2400*(test_id+1)),axis=0)
    deap_data=data_norm(np.delete(deap_data,np.arange(2400*test_id,2400*(test_id+1)),axis=0),resolution)
    
    deap_data_train,deap_label_train=deap_data,deap_label

    deap_data_train=torch.from_numpy(deap_data_train)
    deap_label_train=torch.from_numpy(deap_label_train)
    deap_data_test=torch.from_numpy(deap_data_test)
    deap_label_test=torch.from_numpy(deap_label_test)
    return deap_data_train,deap_data_test,deap_label_train,deap_label_test##deap_data_train：（31*2400） X32X分辨率

def train_and_test_esemble(test_id,emotion_dim,smooth_flag): ## MsDANN: esemble the prediction of models under different resolution
    BATCH_SIZE = 48
    _,hci_data_test_1,_,hci_label_test=get_dataset(50,test_id,emotion_dim,smooth_flag)
    _,hci_data_test_05,_,hci_label_test=get_dataset(91,test_id,emotion_dim,smooth_flag)
    _,hci_data_test_025,_,hci_label_test=get_dataset(173,test_id,emotion_dim,smooth_flag)
    torch_dataset_test=Data.TensorDataset(hci_data_test_1,hci_data_test_05,hci_data_test_025,hci_label_test)
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
            )
    classifier1=DE_mutiscale_classifier_esemble_twoclass(32,50).cuda()
    classifier1.apply(weigth_init)
    classifier2=DE_mutiscale_classifier_esemble_twoclass(32,91).cuda()
    classifier2.apply(weigth_init)
    classifier3=DE_mutiscale_classifier_esemble_twoclass(32,173).cuda()
    classifier3.apply(weigth_init)
    if emotion_dim==0:
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_1_deap')
        classifier1.load_state_dict(torch.load('bestclassifier_'+str(test_id)+'_resolution_0_25'+'.pkl'))
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point5_deap')
        classifier2.load_state_dict(torch.load('bestclassifier_'+str(test_id)+'_resolution_0_25'+'.pkl'))
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point25_deap')
        classifier3.load_state_dict(torch.load('bestclassifier_'+str(test_id)+'_resolution_0_25'+'.pkl'))
    if emotion_dim==1:
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_1_deap')
        classifier1.load_state_dict(torch.load('bestclassifier_arousal_'+str(test_id)+'_resolution_0_25'+'.pkl'))
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point5_deap')
        classifier2.load_state_dict(torch.load('bestclassifier_arousal_'+str(test_id)+'_resolution_0_25'+'.pkl'))
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point25_deap')
        classifier3.load_state_dict(torch.load('bestclassifier_arousal_'+str(test_id)+'_resolution_0_25'+'.pkl'))
    if emotion_dim==2:
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_1_deap')
        classifier1.load_state_dict(torch.load('bestclassifier_dominance_'+str(test_id)+'_resolution_0_25'+'.pkl'))
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point5_deap')
        classifier2.load_state_dict(torch.load('bestclassifier_dominance_'+str(test_id)+'_resolution_0_25'+'.pkl'))
        os.chdir('F:\\zhourushuang\\Multi-Scale\\model_for_poster\\resolution_point25_deap')
        classifier3.load_state_dict(torch.load('bestclassifier_dominance_'+str(test_id)+'_resolution_0_25'+'.pkl'))
    acc,predict_mat,predict_mat_smooth,truth=test_model_esemble_smooth(loader_test,classifier1,classifier2,classifier3)
    result_list={'acc':acc,'p':predict_mat,'p_s':predict_mat_smooth,'t':truth}
    print(acc)
    return result_list

def moving_averge(time_series):
    for k in range(len(time_series)):
        if k>1 and k<58:
            time_series[k]=np.sum(time_series[np.arange(k-2,k+2)],axis=0)/5
    return time_series

def main(resolution,emotion_dim,max_iter,smooth_flag):
    acc_mat=np.zeros((32,1))
    for i in range(32):
        acc_mat[i][0]=train_and_test_GAN(i,resolution,emotion_dim,max_iter,smooth_flag)
    return acc_mat

def main_esemble(emotion_dim,smooth_flag):
    result=[]
    for i in range(32):
        result.append(train_and_test_esemble(i,emotion_dim,smooth_flag))
    return result

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def weight_movingaverage(interval, window_size):
    window= (np.arange(0,window_size)+1)/float(window_size*(window_size+1)/2)
    return np.convolve(interval, window, 'same')

def hull_movingaverage(interval, window_size):## window_size should be a multiple of 2
    series1=weight_movingaverage(interval, window_size/2)*2
    series2=weight_movingaverage(interval, window_size)
    series=series1-series2
    series_smooth=weight_movingaverage(series, int(np.sqrt(window_size)))
    return series_smooth

def smooth(code_mat,smooth_method,resolution): ## smooth the feature using differernt smoothing methods
    flag=0
    code_mat_sque=np.reshape(code_mat,(np.shape(code_mat)[0],32*resolution))
    for z in range(32):
        for i in range(40):
            video_signal=code_mat_sque[flag:flag+60,:]
            for j in range(32*resolution):
                if smooth_method=="standard":
                    video_signal[:,j]=movingaverage(video_signal[:,j],20)    
                if smooth_method=="weighted":
                    video_signal[:,j]=weight_movingaverage(video_signal[:,j],20)
                if smooth_method=="hull":
                    video_signal[:,j]=hull_movingaverage(video_signal[:,j],20)
            code_mat_sque[flag:flag+60,:]=video_signal
            flag+=60
    code_mat=np.reshape(code_mat_sque,(np.shape(code_mat)[0],32,resolution))
    return code_mat

def data_norm(hci_data,resolution):## function for data normalization
    hci_data_sque=np.reshape(hci_data,(np.shape(hci_data)[0],32*resolution))
    hci_data_sque_scaled=max_min_scale(hci_data_sque)
    hci_data=np.reshape(hci_data_sque_scaled,(np.shape(hci_data)[0],32,resolution))
    return hci_data

acc_mat_1=main(50,1,1000,1)

acc_mat_2=main(91,1,1000,1)

acc_mat_3=main(173,1,1000,1)

acc_mat_1=main(50,0,1000,1)

acc_mat_2=main(91,0,1000,1)

acc_mat_3=main(173,0,1000,1)

result_arousal=main_esemble(1,1)

result_valence=main_esemble(0,1)

predict_valence_tran_smooth=[]
predict_valence_tran=[]
truth_valence_tran  =[]
for i in range(32):
    predict_valence_tran_smooth.append(result_valence[i]['p_s'])
    predict_valence_tran.append(result_valence[i]['p'])
    truth_valence_tran.append(result_valence[i]['t'])
predict_valence_tran_smooth=np.vstack(predict_valence_tran_smooth)
predict_valence_tran=np.vstack(predict_valence_tran)
truth_valence_tran=np.vstack(truth_valence_tran)

predict_arousal_tran_smooth=[]
predict_arousal_tran=[]
truth_arousal_tran  =[]
for i in range(32):
    predict_arousal_tran_smooth.append(result_arousal[i]['p_s'])
    predict_arousal_tran.append(result_arousal[i]['p'])
    truth_arousal_tran.append(result_arousal[i]['t'])
predict_arousal_tran_smooth=np.vstack(predict_arousal_tran_smooth)
predict_arousal_tran=np.vstack(predict_arousal_tran)
truth_arousal_tran=np.vstack(truth_arousal_tran)

