from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
#import networkx
#import itertools
import random


FORBID = {
    'train':{
        'freeway': ['10066', '10083', '10103', '10108', '10128', '10158', '10160', '10206' ,'10257', '10290', '10311'],
        'road': ['10368']
    },
    'test':{
        'freeway': ['30110'],
        'road': ['20044', '20101', '20131']
    }
}
class OursDataset(Dataset):
    def __init__(self, data_path, dad_data_path, data_class, feature, aug_type_num, phase='train', ptm_dataset='imagenet', box_type='od',toTensor=False, device=torch.device('cuda'), vis=False, toa_modify=0, all50=True):
        self.data_path = data_path
        self.data_class = data_class
        self.feature = feature
        self.phase = phase
        self.ptm_dataset = ptm_dataset
        self.box_type = box_type
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.all50 = all50
        self.aug_type_num = aug_type_num
        self.n_frames = 60
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)
        self.files_dict, self.labels_dict = self.read_datalist(data_path, phase)
        if phase == 'train':
            self.dad_files_dict, self.dad_labels_dict = self.read_dad_datalist(dad_data_path, phase)
            self.files_dict.update(self.dad_files_dict)
            self.labels_dict.update(self.dad_labels_dict)
        self.name_list = list(self.files_dict.keys())

        self.toa_dict = self.get_toa_all(data_path)
        if phase == 'train':
            dad_toa_dict = self.get_dad_toa_all(dad_data_path, phase)
            self.toa_dict.update(dad_toa_dict)
        self.toa_modify = toa_modify

    def __len__(self):
        data_len = len(self.name_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError
        
    def read_dad_datalist(self, dad_data_path, phase):
        dir_file_path = os.path.join(dad_data_path, phase)
        files = os.listdir(dir_file_path)
        dad_data_files = {}
        dad_data_labels = {}
        for file in files:
            filename = file[:-4]
            dad_data_files[filename] = os.path.join(dir_file_path, file)
            label = int(np.load(os.path.join(dir_file_path, file))['labels'][1])
            dad_data_labels[filename] = label
        
        return dad_data_files, dad_data_labels

    def read_datalist(self, data_path, phase):
        # load training set
        forbid_file = FORBID[phase][self.data_class]
        # list_file = os.path.join(data_path, self.feature + '_features', f'{self.data_class}_{phase}.csv')
        if self.feature == 'res101':
            raise NotImplementedError
            # list_file = os.path.join(data_path, self.feature+'_features_1', self.phase, f'{self.data_class}_{phase}.csv')
        else:
            list_file = f'/data/yehj/SAVES/DSTA/{self.data_class}_{self.phase}.csv'
            # list_file = os.path.join(data_path, f'{self.feature}_features_{self.ptm_dataset}_{self.box_type}_0', self.phase, f'{self.data_class}_{phase}.csv')
        assert os.path.exists(list_file), "file not exists: %s"%(list_file)
        fid = open(list_file, 'r')
        data_files, data_labels = {}, {}
        for line in fid.readlines()[1:]: #Skip first line
            filename, label = line.rstrip().split(',')
            if filename in forbid_file:
                continue
            # ================================= get abs path ===============================
            for aug_idx in range(self.aug_type_num):
                if filename not in data_files.keys():
                    data_files[filename] = []
                abs_path = os.path.join(data_path, f'{self.feature}_features_{self.ptm_dataset}_{self.box_type}_{aug_idx}', self.phase, self.data_class, filename + '.npz')
                data_files[filename].append(abs_path)
            # ==============================================================================
            # data_files.append(filename)
            # if filename not in data_labels.keys():
            #     data_labels[filename] = []
            data_labels[filename] = int(float(label))
        fid.close()
        return data_files, data_labels

    def get_toa_all(self, data_path):
        toa_dict = {}
        annofile = f'/data/yehj/SAVES/DSTA/{self.data_class}_{self.phase}.csv'
        # annoData = self.read_anno_file(annofile)
        # for anno in annoData:
        with open(annofile, 'r') as f:
            for line in f.readlines()[1:]: #Skip first line
                filename, label = line.rstrip().split(',')
                toa_dict[filename] = max(int(label), 1)
        return toa_dict
    
    def get_dad_toa_all(self, dad_data_path, phase):
        dad_toa_dict = {}
        dir_file_path = os.path.join(dad_data_path, phase)
        files = os.listdir(dir_file_path)
        for file in files:
            id = str(np.load(os.path.join(dir_file_path, file))['ID'])
            label = int(np.load(os.path.join(dir_file_path, file))['labels'][1])
            if id in dad_toa_dict.keys():
                print("duplicate key")
                assert 0
            if label==1:
                dad_toa_dict[id+"_"+str(label)] = 50
            else:
                dad_toa_dict[id+"_"+str(label)] = 61

        return dad_toa_dict

    # def read_anno_file(self, anno_file):
    #     assert os.path.exists(anno_file), "Annotation file does not exist! %s"%(anno_file)
    #     result = []
    #     with open(anno_file, 'r') as f:
    #         for line in f.readlines():
    #             items = {}
    #             items['vid'] = line.strip().split(',[')[0]
    #             labels = line.strip().split(',[')[1].split('],')[0]
    #             items['label'] = [int(val) for val in labels.split(',')]
    #             assert sum(items['label']) > 0, 'invalid accident annotation!'
    #             others = line.strip().split(',[')[1].split('],')[1].split(',')
    #             items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
    #             result.append(items)
    #     f.close()
    #     return result

    def __getitem__(self, index):
        # ================================================= no need to osp ===============================================
        # if self.feature == 'res101':
        #     augument_class = random.randint(0, 1)
        #     data_file = os.path.join(self.data_path, self.feature + f'_features_{augument_class}', self.phase, self.data_class, self.files_list[index] + '.npz')
        # else:
        #     data_file = os.path.join(self.data_path, f'{self.feature}_features_{self.ptm_dataset}_{self.box_type}', self.phase, self.data_class, self.files_list[index] + '.npz')
        # assert os.path.exists(data_file), "file not exists: %s"%(data_file)
        data_file_name = self.name_list[index]
        aug_type_num = len(self.files_dict[data_file_name])
        if isinstance(self.files_dict[data_file_name], str):
            data_file = self.files_dict[data_file_name] # for dad dataset
        else:
            aug_idx = random.randint(0, aug_type_num - 1)
            data_file = self.files_dict[data_file_name][aug_idx]
        try:
            data = np.load(data_file)
            features = data['data']
            if 'label' in data.files:
                labels = data['label']  # 2
            else:
                labels = data['labels']
            detections = data['det']
            vid = str(data['ID'])
        except:
            raise IOError('Load data error! File: %s'%(data_file))
        

        if len(vid) == 9:
            features = features[: -10, :, :]  # cut last 10 frames in dad
            detections = detections[: -10, :, :]
        if detections.shape[0] > 60:
            features = features[detections.shape[0] - 60:, :, :]  # only for last 50 frames. 50 x 20 x 4096
            detections = detections[detections.shape[0] - 60:, :, :]  # 50 x 19 x 6   
        elif detections.shape[0] < 60:
            features_add = np.repeat(features[-1:,:,:], 60 - detections.shape[0], axis=0)    # repeat the last frame
            detections_add = np.repeat(detections[-1:,:,:], 60 - detections.shape[0], axis=0)
            
            features = np.concatenate((features, features_add), axis=0)
            detections = np.concatenate((detections, detections_add), axis=0)
        
        if vid not in self.toa_dict.keys():
            vid = vid + "_" + str(int(labels[1]))
        
            
        toa = []
        if labels[1] > 0:
            if self.all50:
                toa.append(50)
            else:
                if self.toa_dict[vid]+self.toa_modify < 55 and self.toa_dict[vid]+self.toa_modify > 5:
                    toa.append(self.toa_dict[vid]+self.toa_modify)
                elif self.toa_dict[vid]+self.toa_modify <= 5:
                    toa.append(5)
                elif self.toa_dict[vid]+self.toa_modify >= 55:
                    toa.append(55)
            # toa = [self.toa_dict[vid]+self.toa_modify if self.toa_dict[vid]+self.toa_modify <= 60 else 60]
        else:
            toa = [self.n_frames + 1]

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)         #  50 x 20 x 4096
            labels = torch.Tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            return features, labels, toa, detections, vid
        else:
            return features, labels, toa


class DADDataset(Dataset):
    def __init__(self, data_path, feature, phase='training', toTensor=False, device=torch.device('cuda'), vis=False):
        # self.data_path = os.path.join(data_path, feature + '_features')
        self.data_path = data_path
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 100
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)

        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s"%(filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list
    
    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['data']  # 100 x 20 x 4096
            labels = data['labels']  # 2
            detections = data['det']  # 100 x 19 x 6
        except:
            raise IOError('Load data error! File: %s'%(data_file))
        # print(f'label:{labels[1]}')
        toa = []
        for i in range(labels.shape[0]):
            if labels[i, 1] > 0:
                toa.append(90.0)
            else:
                toa.append(self.n_frames + 1)

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)         #  100 x 20 x 4096
            labels = torch.Tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        # print("features:", features.shape)
        # print("labels:", labels.shape)
        # print("toa:", toa.shape)
        # assert 0

        if self.vis:
            video_id = str(data['ID'])[5:11]  # e.g.: b001_000490_*
            return features, labels, toa, detections, video_id
        else:
            return features, labels, toa
