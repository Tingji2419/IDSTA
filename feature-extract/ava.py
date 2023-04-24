import os
import json
import numpy as np
import torch
import platform

from random import randint
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import csv

from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET


class AVA(Dataset):
    def __init__(self, dataloader_type, data_class, box_type='original') -> None:
        super().__init__()
        self.data_path = '/data/yehj/SAVES/DSTA'
        self.dataloader_type = dataloader_type
        self.data_class = data_class
        self.box_type = box_type

        self.path_list = []

        self.get_data_file_path()
        self.get_video_label()
        
        self.video_info = {}
        self.frame_label = {}
        


    def get_files_in_folder(self, folder_path,path_type='file',file_type='jpg'):
        return_list = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if path_type == 'file':
                if os.path.isfile(item_path):
                    return_list.append(item_path)
            elif path_type == 'dir':
                if os.path.isdir(item_path):
                    return_list.append(item_path)

        return_list.sort()

        if path_type == 'dir' and file_type == 'jpg':
            self.path_list = self.path_list + return_list

        return return_list
    

    def get_data_file_list(self, dir_path):
        # get folder name
        jpg_dir_path = os.path.join(dir_path, 'jpgs')
        xml_dir_path = os.path.join(dir_path, f'xmls_{self.box_type}')
        jpg_dir_path_list = self.get_files_in_folder(jpg_dir_path,path_type='dir')
        xml_dir_path_list = self.get_files_in_folder(xml_dir_path,path_type='dir',file_type='xml')

        # get file name list
        jpg_file_path_list = []
        xml_file_path_list = []

        for jpg_dir,xml_dir in zip(jpg_dir_path_list, xml_dir_path_list):
            jpg_file_path_list.append(self.get_files_in_folder(jpg_dir))
            xml_file_path_list.append(self.get_files_in_folder(xml_dir,file_type='xml'))

        return jpg_file_path_list, xml_file_path_list
    

    def get_data_file_path(self):
        dir_path_list = []
        if self.dataloader_type=='train':
            train_path = os.path.join(self.data_path, "train_dataset")
            if self.data_class=='freeway':
                dir_path_list.append(os.path.join(train_path, 'freeway_train'))
            elif self.data_class=='road':
                dir_path_list.append(os.path.join(train_path, 'road_train'))
            elif self.data_class=='total':
                dir_path_list.append(os.path.join(train_path, 'freeway_train'))
                dir_path_list.append(os.path.join(train_path, 'road_train'))
            else:
                assert 0

        elif self.dataloader_type=='test':
            test_path = os.path.join(self.data_path, "test_dataset")
            if self.data_class=='freeway':
                dir_path_list.append(os.path.join(test_path, 'freeway_test'))
            elif self.data_class=='road':
                dir_path_list.append(os.path.join(test_path, 'road_test'))
            elif self.data_class=='total':
                dir_path_list.append(os.path.join(test_path, 'freeway_test'))
                dir_path_list.append(os.path.join(test_path, 'road_test'))
            else:
                assert 0

        
        self.video_img_path_list = []
        self.video_xml_path_list = []

        for dir_path in dir_path_list:
            img_path, xml_path = self.get_data_file_list(dir_path)
            self.video_img_path_list = self.video_img_path_list + img_path
            self.video_xml_path_list = self.video_xml_path_list + xml_path

        
    def read_video(self, index):
        image_paths = self.video_img_path_list[index]
        image_paths.sort()

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                # image_tensor = torch.Tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0)
                images.append(np.array(image))

        # 在新的维度上进行拼接
        # images_tensor = torch.cat(images, dim=0)
        image_array = np.stack(images, axis=0)

        return image_array
    
    
    def get_data_vaule(self, root, style, typename, typevalue, valuename):
        nodelist = root.getElementsByTagName(style) # 根据标签的名字获得节点列表

        for node in nodelist: 
            if typevalue == node.getAttribute(typename):
                node_name = node.getElementsByTagName(valuename)
                value = node_name[0].childNodes[0].nodeValue
                return value
    

    def extract_column_from_xml(self, xml_file, frame_idx, xml_info_list):
        # 加载XML文件

        # tree = ET.parse(xml_file)
        # root = tree.getroot()

        # new_dict = {}

        # # print(width, height)
        # for i, obj in enumerate(root.iter('object')):
        #     obj_name = obj.find('name').text
        #     # obj_name = obj_name + f"_{i}"
        #     xml_box = obj.find('bndbox')
        #     xmin = (int(xml_box.find('xmin').text)) 
        #     ymin = (int ( xml_box.find('ymin').text)) 
        #     xmax = (int(xml_box.find('xmax').text)) 
        #     ymax = (int(xml_box.find('ymax').text)) 

        #     if obj_name not in new_dict.keys():
        #         new_dict[obj_name] = (xmin,ymin,xmax,ymax)

        # xml_info_list.append(new_dict)
        new_dict = {}
        with open(xml_file, 'r') as f:
            tmp = f.readlines()
        for cur_object in tmp[:19]: # Warning for 21 boxes
            cur_object = cur_object.strip().split(',')
            obj_name = cur_object[0]
            xmin = (int(float(cur_object[1]))) 
            ymin = (int(float(cur_object[2]))) 
            xmax = (int(float(cur_object[3]))) 
            ymax = (int(float(cur_object[4]))) 
            if obj_name not in new_dict.keys():
                new_dict[obj_name] = (xmin, ymin, xmax, ymax)
        xml_info_list.append(new_dict)
        return xml_info_list

    def read_excel(self, index):
        xml_paths = self.video_xml_path_list[index]
        xml_paths.sort()
        video_info = []

        for i, xml_path in enumerate(xml_paths):
            video_info = self.extract_column_from_xml(xml_path, i, video_info)

        return video_info
    

    def get_video_label(self):
        csv_path_list = []
        if self.dataloader_type=='train':
            train_path = os.path.join(self.data_path, "train_dataset")
            if self.data_class=='freeway':
                csv_path_list.append(os.path.join(train_path, 'freeway_train.csv'))
            elif self.data_class=='road':
                csv_path_list.append(os.path.join(train_path, 'road_train.csv'))
            elif self.data_class=='total':
                csv_path_list.append(os.path.join(train_path, 'freeway_train.csv'))
                csv_path_list.append(os.path.join(train_path, 'road_train.csv'))
            else:
                assert 0

        elif self.dataloader_type=='test':
            train_path = os.path.join(self.data_path, "test_dataset")
            if self.data_class=='freeway':
                csv_path_list.append(os.path.join(train_path, 'freeway_test.csv'))
            elif self.data_class=='road':
                csv_path_list.append(os.path.join(train_path, 'road_test.csv'))
            elif self.data_class=='total':
                csv_path_list.append(os.path.join(train_path, 'freeway_test.csv'))
                csv_path_list.append(os.path.join(train_path, 'road_test.csv'))
            else:
                assert 0

        self.video_label_dict = {}
        for csv_path in csv_path_list:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                headers = next(reader)  # 跳过第一行
                for row in reader:
                    self.video_label_dict[row[0]] = int(float(row[1]))
                    # print(row)

    def return_ins_num(self):
        return len(self.video_img_path_list)
    

    def get_excel(self, index):
        video_name = self.path_list[index].split('/')[-1].split(".")[0]
        video_info = self.read_excel(index)

        frame_num = len(video_info)

        xml_paths = self.video_xml_path_list[index]
        xml_paths.sort()

        xml_name_list = []
        xml_info_list = []

        for i in range(frame_num):
            xml_name_list.append(xml_paths[i])
            xml_info_list.append(video_info[i])
        
        
        return xml_name_list, xml_info_list

    def get_frame_label(self, video_info, video_label, frame_len):  # 输入为1，输出为60
        if video_label == 0:
            frame_label = np.zeros(frame_len)
        
        else:
            frame_label = np.ones(frame_len)
        #     try:
        #         if not self.only_end:
        #             mid_frame_of_acc_car = []
        #             for key in video_info.keys():
        #                 # print("video_info.keys():", video_info.keys())
        #                 if 'acc' in key:
        #                     mid_frame_of_acc_car.append(video_info[key]['start_idx'] + len(video_info[key]['bbox_info'])/2)
                    
        #             acc_start_frame = int((max(mid_frame_of_acc_car) + min(mid_frame_of_acc_car)) / 2)

        #         else:
        #             acc_start_frame = 49
                
        #         # 创建全0向量和全1向量
        #         frame_label_zero = np.zeros(acc_start_frame)
        #         frame_label_one = np.ones((1, frame_len - len(frame_label_zero)))

        #         # 将两个向量拼接在一起
        #         frame_label = np.concatenate((frame_label_zero, frame_label_one), axis=None)
                
        #     except:
        #         if not self.only_end:
        #             frame_label_zero = np.zeros(30)
        #             frame_label_one = np.ones((1, frame_len - 30))
                    
        #             # 将两个向量拼接在一起
        #             frame_label = np.concatenate((frame_label_zero, frame_label_one), axis=None)

        #         else:
        #             frame_label_zero = np.zeros(49)
        #             frame_label_one = np.ones((1, frame_len - 49))
                    
        #             # 将两个向量拼接在一起
        #             frame_label = np.concatenate((frame_label_zero, frame_label_one), axis=None)


        return frame_label
    

    def __getitem__(self, index):
        # for index in range(363):
        video_name = self.path_list[index].split('/')[-1].split(".")[0]

        # read video img
        video_data = self.read_video(index)     # [T, C, H, W]

        # read video xml info
        if video_name not in self.video_info.keys():
            video_info = self.read_excel(index)
            self.video_info[video_name] = video_info
        else:
            video_info = self.video_info[video_name]

        # get video label 
        video_label = self.video_label_dict[video_name]

        # get frame label
        if video_name not in self.frame_label.keys():
            frame_len = video_data.shape[0]
            frame_label = self.get_frame_label(video_info, video_label, frame_len)
            self.frame_label[video_name] = frame_label
        else:
            frame_label = self.frame_label[video_name]

        # video_data = self.transform['image'](video_data)

        return video_data, frame_label, video_info, video_name

    
    def __len__(self):
        return len(self.video_img_path_list)
    

if __name__ == "__main__":
    dataloader_type = 'train'
    data_class = 'freeway'

    dataset = AVA(dataloader_type, data_class)
    train_dataloader = DataLoader(dataset=dataset, batch_size=1)
    for index, (images, labels, info) in enumerate(train_dataloader):

        print(index)
        print(info)
