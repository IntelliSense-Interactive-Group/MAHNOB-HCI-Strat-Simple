import pandas as pd
import numpy as np
import pyedflib ## 读取BDF数据 
import matplotlib.pyplot as plt

# pytorch相关
import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

# 信号处理
from scipy import signal
from scipy.fftpack import fft,ifft,fftshift
from scipy.signal import welch

import random

# 网络结构
from torchsummary import summary
from torch.autograd import Variable

# python自带工具包
from functools import reduce
from operator import __add__

from sklearn import preprocessing

from model import *

####################################################################################
####################################################################################
##################################### Get Data #####################################

## 统计每个label出现次数
def get_label_dic(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

## 读取BDF文件数据，bdf_file为待读取通道名字，name为待读取通道名字， start和end为读取时间范围
def LoadBDF(bdf_file, name = "EXG2", start = None, end = None):
    with pyedflib.EdfReader(bdf_file) as f:
        status_index = f.getSignalLabels().index('Status')
        sample_frequency = f.samplefrequency(status_index)
        status_size = f.samples_in_file(status_index)
        status = np.zeros((status_size), dtype = 'float64')
        f.readsignal(status_index, 0, status_size, status)
        status = status.round().astype('int')
        nz_status = status.nonzero()[0]

        video_start = nz_status[0]
        video_end = nz_status[-1]

        index = f.getSignalLabels().index(name)
        sample_frequency = f.samplefrequency(index)

        video_start_seconds = video_start / sample_frequency

        if start is not None:
            start += video_start_seconds
            start *= sample_frequency
            if start < video_start:
                start = video_start
            start = int(start)
        else:
            start = video_start
        
        if end is not None:
            end += video_start_seconds
            end *= sample_frequency
            if end > video_end:
                end = video_end
            end = int(end)
        else:
            end = video_end
        
#         PhysicalMax = f.getPhysicalMaximum(index)
#         PhysicalMin = f.getPhysicalMinimum(index)
#         DigitalMax = f.getDigitalMaximum(index)
#         DigitalMin = f.getDigitalMinimum(index)

#         scale_factor = (PhysicalMax - PhysicalMin) / (DigitalMax - DigitalMin)
#         dc = PhysicalMax - scale_factor * DigitalMax

        container = np.zeros((end - start + 1), dtype = 'float64')
        f.readsignal(index, start, end - start + 1, container)
#         container = container * scale_factor + dc

        return container, sample_frequency

def get_eeg_data(path):

#     chan_list = ["Fp1","AF3","F3","F7","FC1","FC5","T7","C3","CP1","CP5","P7","P3","PO3","O1",
#                  "Fp2","AF4","F4","F8","FC2","FC6","T8","C4","CP2","CP6","P8","P4","PO4","O2",
#                 "Fz","Cz","Pz","Oz"]
    chan_list = ["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1",
                 "Oz","Pz","Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","CP6","CP2","P4","P8","PO4","O2"]

    eeg_data = []

    # status, freq = LoadBDF(path,"Status")
    # status_pd = pd.Series(status)
    # valid_index = status_pd[status_pd<status_pd[0]].index

    for i in np.arange(len(chan_list)):
        data, fs1 = LoadBDF(path, name=chan_list[i])
    #     data = data[valid_index]
        data = (data-np.mean(data))/np.std(data)

        eeg_data.append(data)
    
    eeg_data_pd = pd.DataFrame(np.asarray(eeg_data).transpose(1,0))
    eeg_data_pd.columns = ["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1",
                 "Oz","Pz","Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","CP6","CP2","P4","P8","PO4","O2"]
    return eeg_data_pd


def get_eye_track_data(path):
    eye_track_data = pd.read_csv(path, sep="\t", skiprows=23, error_bad_lines=False)

    # data1 = eye_track_data[['GazePointXLeft','GazePointYLeft','CamXLeft','CamYLeft',
    #                         'DistanceLeft','PupilLeft','ValidityLeft','GazePointXRight',
    #                         'GazePointYRight','CamXRight','CamYRight','DistanceRight',
    #                         'PupilRight','ValidityRight','GazePointX','GazePointY',
    #                         'Event']]
    data1 = eye_track_data[['DistanceLeft','PupilLeft','DistanceRight',
                            'PupilRight','Event']]
    data2 = data1.loc[np.asarray(data1[data1['Event']=='MovieStart'].index)[0]:np.asarray(data1[data1['Event']=='MovieEnd'].index)[0]+1]
    # data3 = data2[data2['ValidityLeft']<2]
    # data4 =  data3[data3['ValidityRight']<2]
    data5 = data2.drop(['Event'], axis=1)
    data6 = np.asarray(data5)
    data7 = data6[~np.isnan(data6).any(axis=1), :]
#     data6 = np.asarray(data5)

#     data7 = eye_track_data[['DistanceLeft','PupilLeft','DistanceRight',
#                         'PupilRight','StimuliName']]
#     data8 = data7.iloc[data7[data7['StimuliName'] == '69.avi'].index]
#     data9 = data8.drop(['StimuliName'],axis=1)
    
    return data7


## win_len shoud set = freq
def re_data(data, win_len, overlap):
    step_len = win_len-int(win_len*overlap)
    data_reinf = []
    
    i = 0
    while (i+win_len <= len(data)):
        data_reinf.append(data[i:i+win_len])
        i = i + step_len
    
    return np.asarray(data_reinf)


def get_label(path):
    label_data = pd.read_csv(path, sep="\t")

#     arousal_index = label_data[label_data['Name']=='Arousal assessment'].index
#     valence_index = label_data[label_data['Name']=='Valence assessment'].index
    emotion_index = label_data[label_data['Name']=='Emotion keyword'].index
    emotion_label = []
    for i in np.arange(0,len(emotion_index),2):
        emotion_label.append(label_data.iloc[emotion_index[i]+1]['Name'])
        
    arousal_label_map = []
    valence_label_map = []
    for i in np.arange(len(emotion_label)):
        if (emotion_label[i]=='D1' or emotion_label[i]=='D3' or emotion_label[i]=='D4'):
            arousal_label_map.append(0) ## Calm
        elif (emotion_label[i]=='D2' or emotion_label[i]=='D5'):
            arousal_label_map.append(1) ## Medium arousal
        elif (emotion_label[i]=='D8' or emotion_label[i]=='D7' or emotion_label[i]=='D6' or emotion_label[i]=='D9'):
            arousal_label_map.append(2) ## Excited/Activated

        if(emotion_label[i]=='D7' or emotion_label[i]=='D6' or emotion_label[i]=='D1' or emotion_label[i]=='D9' or emotion_label[i]=='D3'):
            valence_label_map.append(0) ## Unpleasant
        elif(emotion_label[i]=='D8' or emotion_label[i]=='D4'):
            valence_label_map.append(1) ## Neutral valence
        elif(emotion_label[i]=='D2' or emotion_label[i]=='D5'):
            valence_label_map.append(2)  ## pleasant      
    
    return np.asarray(arousal_label_map), np.asarray(valence_label_map)


def get_emotion_label(path):
    label_data = pd.read_csv(path, sep="\t")

    emotion_index = label_data[label_data['Name']=='Emotion keyword'].index
    
    emotion_label = []
    for i in np.arange(0,len(emotion_index),2):
        emotion_label.append(label_data.iloc[emotion_index[i]+1]['Name'])
        
    emotion_label_map = []
    for i in np.arange(len(emotion_label)):
        if (emotion_label[i]=='D1'):
            emotion_label_map.append(0)
        elif (emotion_label[i]=='D2'):
            emotion_label_map.append(1)
        elif (emotion_label[i]=='D3'):
            emotion_label_map.append(2)
        elif (emotion_label[i]=='D4'):
            emotion_label_map.append(3)
        elif (emotion_label[i]=='D5'):
            emotion_label_map.append(4)
        elif (emotion_label[i]=='D6'):
            emotion_label_map.append(5)
        elif (emotion_label[i]=='D7'):
            emotion_label_map.append(6)
        elif (emotion_label[i]=='D8'):
            emotion_label_map.append(7)
        elif (emotion_label[i]=='D9'):
            emotion_label_map.append(8)
    
    return np.asarray(emotion_label_map)

def get_one_data_pd(participant_no):
#     print(participant_no)
    eeg_path = "/mnt/nvme0n1/zhanggaotian/HCI/Sessions/{}/Part_{}_S_Trial{}_emotion.bdf"
    eye_track_path = "/mnt/nvme0n1/zhanggaotian/HCI/Sessions/{}/P{}-Rec1-All-Data-New_Section_{}.tsv"
    label_path = "/mnt/nvme0n1/zhanggaotian/HCI/Sessions/{}/P{}-Rec1-Guide-Cut.tsv"
    
# if participant_no not in [3,9,12,15,16,26, 2,10,25]:
    arousal_label_map, valence_label_map = get_label(label_path.format(str((participant_no-1)*130+1),str(participant_no)))
#     print(label_path.format(str((participant_no-1)*130+1),str(participant_no)))
    emotion_label_map = get_emotion_label(label_path.format(str((participant_no-1)*130+1),str(participant_no)))
    
    eeg_trial_list = []
    eye_track_trial_list = []
    
    for trial in np.arange(2,41,2):
        eeg_trial = get_eeg_data(eeg_path.format(str((participant_no-1)*130+trial), str(participant_no), str(int(trial/2))))
        eye_track_trial = get_eye_track_data(eye_track_path.format(str((participant_no-1)*130+trial), str(participant_no), str(trial)))
        
        eeg_trial_list.append(eeg_trial)
        eye_track_trial_list.append(eye_track_trial)
    return eeg_trial_list, eye_track_trial_list, arousal_label_map, valence_label_map, emotion_label_map


def get_one_data_re(participant_no):
#     print(participant_no)
    eeg_path = "/mnt/nvme0n1/zhanggaotian/HCI/Sessions/{}/Part_{}_S_Trial{}_emotion.bdf"
    eye_track_path = "/mnt/nvme0n1/zhanggaotian/HCI/Sessions/{}/P{}-Rec1-All-Data-New_Section_{}.tsv"
    label_path = "/mnt/nvme0n1/zhanggaotian/HCI/Sessions/{}/P{}-Rec1-Guide-Cut.tsv"
    
# if participant_no not in [3,9,12,15,16,26, 2,10,25]:
    arousal_label_map, valence_label_map = get_label(label_path.format(str((participant_no-1)*130+1),str(participant_no)))


    trial = 2
    eeg_trial = get_eeg_data(eeg_path.format(str((participant_no-1)*130+trial), str(participant_no), str(int(trial/2))))
    eye_track_trial = get_eye_track_data(eye_track_path.format(str((participant_no-1)*130+trial), str(participant_no), str(trial)))

    re_eeg_data = re_data(eeg_trial, 256*10, 0.5)
    re_eye_track_data = re_data(eye_track_trial, 60*10, 0.5)

    if len(re_eeg_data)==len(re_eye_track_data):

        trial_arousal_label = np.repeat(arousal_label_map[int(trial/2)-1],len(re_eye_track_data))
        trial_valence_label = np.repeat(valence_label_map[int(trial/2)-1],len(re_eye_track_data)) 


    for trial in np.arange(4,41,2):
        eeg_trial = get_eeg_data(eeg_path.format(str((participant_no-1)*130+trial), str(participant_no), str(int(trial/2))))
        eye_track_trial = get_eye_track_data(eye_track_path.format(str((participant_no-1)*130+trial), str(participant_no), str(trial)))

        re_eeg_data_loop = re_data(eeg_trial, 256*10, 0.5)
        re_eye_track_data_loop = re_data(eye_track_trial, 60*10, 0.5)

        if len(re_eeg_data_loop)==len(re_eye_track_data_loop):
            re_eeg_data = np.concatenate((re_eeg_data, re_eeg_data_loop))
            re_eye_track_data = np.concatenate((re_eye_track_data, re_eye_track_data_loop))

            trial_arousal_label_loop = np.repeat(arousal_label_map[int(trial/2)-1],len(re_eeg_data_loop))
            trial_valence_label_loop = np.repeat(valence_label_map[int(trial/2)-1],len(re_eeg_data_loop))

#                 print(trial_arousal_label_loop.shape)
#                 print(trial_valence_label_loop.shape)

            trial_arousal_label = np.concatenate((trial_arousal_label, trial_arousal_label_loop))
            trial_valence_label = np.concatenate((trial_valence_label, trial_valence_label_loop))

#                 print(re_eeg_data.shape)
#                 print(re_eye_track_data.shape)
#                 print(trial_arousal_label.shape)
    return re_eeg_data, re_eye_track_data, trial_arousal_label, trial_valence_label


class subDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        return data, label

def weigth_init(m):
    
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.2)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.2)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def create_model(model_name, verbose=True):
    
    if model_name=="Simple_CNN":
        model = Simple_CNN()
        model.apply(weigth_init)
        if torch.cuda.is_available():
            model = model.cuda()
            if verbose:
                summary(model, (32, 1, 2560))
                print(model)
        return model
    
    if model_name=="CNNModel":
        model = CNNModel()
        model.apply(weigth_init)
        if torch.cuda.is_available():
            model = model.cuda()
            if verbose:
                summary(model, (32, 1, 2560))
                print(model)
        return model
    
    return None


def filter_data(low, high, data, fs=250):
    
    b, a = signal.butter(4, [2*low/fs,2*high/fs], 'bandpass')
    filter_data = []
    for chan_index in np.arange(data.shape[1]):
        data_f = signal.filtfilt(b, a, data[:,chan_index])
        filter_data.append(data_f)
        
    return np.asarray(filter_data).T


## notch_freq陷波频率 
## Q陷波质量
def filter_data_notch(notch_freq, Q, data, fs=250):   

    w0 = notch_freq/(fs/2)
    b, a = signal.iirnotch(w0=w0, Q=Q)
    filter_data = []
    for chan_index in np.arange(data.shape[1]):
        data_f = signal.filtfilt(b, a, data[:,chan_index])
        filter_data.append(data_f)
    
    return np.asarray(filter_data).T


def get_fbp(data, fs, fmin, fmax):
    
    f, Pxx = welch(data.T, fs=fs, nperseg = 250)
    
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1

    return Pxx[:,ind_min:ind_max].sum(1)

def re_data2(trials, labels, wid_len, step_len):
    
    new_trials = []
    new_labels = []
    trial_index = 0
    min_max_scaler = preprocessing.MinMaxScaler()
    
    for trial in trials:
        for wid_index in np.arange((len(trial)-wid_len)//step_len):
#             print(trial[wid_index*wid_len:(wid_index+1)*wid_len,:].shape)
            new_trial = trial[wid_index*step_len:wid_index*step_len+wid_len,:]
            new_trial_filter = filter_data(0.5,50,new_trial)
            new_trial_notch = filter_data_notch(50,5,new_trial_filter)
            new_trials.append(min_max_scaler.fit_transform(new_trial_notch))
            new_labels.append(labels[trial_index])
        trial_index = trial_index+1
    return np.asarray(new_trials), np.asarray(new_labels)

def balance_data_and_re(eeg_trial_list, labels, window_len, overlap, split_rate):
    
    label0 = np.where(labels==0)[0]
    label1 = np.where(labels==1)[0]
    label2 = np.where(labels==2)[0]
    
    eeg_trials = []
    for i in np.arange(len(eeg_trial_list)):
        eeg_trials.append(np.asarray(eeg_trial_list[i]))
        
    label0_len = 0
    label1_len = 0
    label2_len = 0
    trials0 = []
    trials1 = []
    trials2 = []
    for i in np.arange(len(eeg_trials)):
        if i in label0:
            label0_len = label0_len + eeg_trials[i].shape[0]
            trials0.append(eeg_trials[i])
        elif i in label1:
            label1_len = label1_len + eeg_trials[i].shape[0]
            trials1.append(eeg_trials[i])
        else:
            label2_len = label2_len + eeg_trials[i].shape[0]
            trials2.append(eeg_trials[i])
    label_len = label0_len+label1_len+label2_len
    label_percent = (label0_len/label_len,label1_len/label_len,label2_len/label_len)
    
    step1_len = int(window_len*(1-overlap))
    step2_len = int(step1_len/label_percent[0]*label_percent[1])
    step3_len = int(step1_len/label_percent[0]*label_percent[2])
    
    new_trials0,new_labels0 = re_data2(trials0, labels[label0], window_len, step1_len)
    new_trials1,new_labels1 = re_data2(trials1, labels[label1], window_len, step2_len)
    new_trials2,new_labels2 = re_data2(trials2, labels[label2], window_len, step3_len)
    
    train_x = np.concatenate([new_trials0[:int(len(new_trials0)*split_rate),:,:],new_trials1[:int(len(new_trials1)*split_rate),:,:],new_trials2[:int(len(new_trials2)*split_rate),:,:]])
    train_y = np.concatenate([new_labels0[:int(len(new_labels0)*split_rate)],new_labels1[:int(len(new_labels1)*split_rate)],new_labels2[:int(len(new_labels2)*split_rate)]])
    test_x = np.concatenate([new_trials0[int(len(new_trials0)*split_rate):,:,:],new_trials1[int(len(new_trials1)*split_rate):,:,:],new_trials2[int(len(new_trials2)*split_rate):,:,:]])
    test_y = np.concatenate([new_labels0[int(len(new_labels0)*split_rate):],new_labels1[int(len(new_labels1)*split_rate):],new_labels2[int(len(new_labels2)*split_rate):]])
    
    train_x = train_x[:,np.newaxis,:,:].transpose(0,3,1,2)
    train_y = train_y.reshape(-1,1)
    test_x = test_x[:,np.newaxis,:,:].transpose(0,3,1,2)
    test_y = test_y.reshape(-1,1)
    
    return train_x, train_y, test_x, test_y


def my_norm_data(trial_data):
    trial_norm_result = []
    for i in np.arange(trial_data.shape[1]):
        trial_norm_result.append(trial_data[:,i])
    return np.asarray(trial_norm_result)

def split_train_test(re_eeg_data, labels, split_rate):
    
    norm_re_eeg_data = []
    for i in np.arange(len(re_eeg_data)):
        norm_re_eeg_data.append(my_norm_data(re_eeg_data[i]))
    re_eeg_data = np.asarray(norm_re_eeg_data)
    
    random.seed(10)
    
    label0_index = np.asarray(random.sample(list(np.where(labels==0)[0]), int((labels==0).sum()*(1-split_rate))))
    label1_index = np.asarray(random.sample(list(np.where(labels==1)[0]), int((labels==1).sum()*(1-split_rate))))
    label2_index = np.asarray(random.sample(list(np.where(labels==2)[0]), int((labels==2).sum()*(1-split_rate))))
    
    test_index = np.concatenate([label0_index,label1_index,label2_index])
    train_index = np.setdiff1d(np.arange(len(labels)),test_index)
    
    train_x = re_eeg_data[train_index][:,:,np.newaxis,:]
    train_y = labels[train_index].reshape(-1,1)

    test_x = re_eeg_data[test_index][:,:,np.newaxis,:]
    test_y = labels[test_index].reshape(-1,1)
    
    return train_x, train_y, test_x, test_y

####################################################################################
####################################################################################
############################# Get Time Series Features #############################

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel 部分统计学特征
def get_statistical_feature(data):
    data_std = data.std(axis=0)
    data_mean = data.mean(axis=0)
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    data_percentile = np.percentile(data, [25, 50, 75], axis=0).flatten()
#     data_percentile_25 = np.percentile(data, [25, 50, 75], axis=0)[0]
#     data_percentile_50 = np.percentile(data, [25, 50, 75], axis=0)[1]
#     data_percentile_75 = np.percentile(data, [25, 50, 75], axis=0)[2]
    data_negative_rate = (data<0).sum(axis=0)/len(data)
    return np.concatenate((data_std,data_mean,data_max,data_min,data_percentile,data_negative_rate),axis=0)

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel一阶差分绝对值的平均值
def first_order_difference(data):
    N = len(data)
    tmp = 0
    for i in np.arange(N-1):
        tmp += np.abs(data[i+1]-data[i])
    res = tmp/N
    return res

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel二阶差分绝对值的平均值
def second_order_difference(data):
    N = len(data)
    tmp = 0
    for i in np.arange(N-2):
        tmp += np.abs(data[i+2]-data[i])
    res = tmp/(N-1)
    return res

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel归一化的一阶差分
## 使用该函数时保证前面有求一阶差分绝对值的平均值函数即first_order_difference函数存在
def norm_first_order_difference(data):
    return first_order_difference(data)/np.std(data,axis=0)

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel归一化的二阶差分
## 使用该函数时保证前面有求二阶差分绝对值的平均值函数即second_order_difference函数存在
def norm_second_order_difference(data):
    return second_order_difference(data)/np.std(data,axis=0)

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel的时域能量
def get_engery(data):
    N = len(data)
    tmp = 0
    for i in np.arange(N):
        tmp += (data[i])*(data[i])
    return tmp

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel的功率
def get_power(data):
    return get_engery(data)/len(data)

## data：可以是一个trail数据，也可以是单个通道数据
## return: trail/channel的Hjorth参数特征-activity
def get_hjorth_activity(data):
    N = len(data)
    tmp = 0
    avg_s = np.average(data,axis=0)
    for i in np.arange(N):
        tmp += ((data[i])-avg_s)*((data[i])-avg_s)
    res = tmp/N
    return res

## data：单个trail数据
## return: trail的Hjorth参数特征-mobility&complexity
def get_hjorth_mobility_complexity(data):
    D = np.diff(data,axis=0)
    D = np.insert(D,0,0,axis=0)
    
    N = len(data)
    
    M2 = np.sum(D ** 2, axis = 0) / N
    TP = np.sum(data ** 2, axis = 0)
    M4 = 0
    for i in range(N-1):
        M4 += (D[i+1] - D[i]) ** 2
    M4 = M4 / N
    
    mobility = np.sqrt(M2 / TP)
    complexity = np.sqrt(M4 * TP / M2 / M2)
    
    return mobility, complexity

## data：单个trail数据
## return: trail的高阶过零分析(HOC)统计量
def get_hoc(data):
    nzc = []
    for i in range(10):
        curr_diff = np.diff(data, n=i)

        x_t = curr_diff >= 0   # binary time series signal
        x_t = np.diff(x_t)  # taking diff of x_t
        x_t = np.abs(x_t)   # taking abs value

        count = np.count_nonzero(x_t)
        nzc.append(count)
    return nzc

def get_other_time_sereis_features(data):
    fod = first_order_difference(data)
    sod = second_order_difference(data)
    nfod = norm_first_order_difference(data)
    nsod = norm_second_order_difference(data)
    egy = get_engery(data)
    pwr = get_power(data)
    act = get_hjorth_activity(data)
    mob, compl = get_hjorth_mobility_complexity(data)
    hoc = get_hoc(data)
    return np.concatenate((fod, sod, nfod, nsod, egy, pwr, act, mob, compl, hoc), axis=0)