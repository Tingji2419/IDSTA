#!/usr/bin/env python
# coding: utf-8
'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse
import shutil

from torch.utils.data import DataLoader
# from src.Models_softlabel import DSTA
from src.Models import IDSTA, DSTA
# import ipdb
# import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from tqdm import tqdm

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)
import torch.backends.cudnn as cudnn
import random
np.random.seed(seed=seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

def average_losses(losses_all):
    total_loss, cross_entropy, aux_loss = 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        aux_loss += losses['auxloss']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['auxloss'] = aux_loss / len(losses_all)
    return losses_mean



def test_all_vis(testdata_loader, model, vis=True, multiGPU=False, device=torch.device('cuda')):

    if multiGPU:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.eval()


    all_pred = []
    all_mil_pred = []
    all_labels = []
    all_toas = []
    vis_data = []
    all_ids = []

    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas, detections, video_ids) in tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            # run forward inference

            
            losses, all_outputs, mil_pred, hiddens, alphas = model(batch_xs, batch_ys, batch_toas,
                hidden_in=None,  nbatch=len(testdata_loader), testing=False)


            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            # all outputs:: [60, batchsize, 2]
            # run inference
            all_outputs = torch.stack(all_outputs)
            soft_max = False
            if soft_max:
                for cur_video in range(batch_size):
                    pred_frames[cur_video, :] = np.array(torch.softmax(all_outputs[:, cur_video], dim=1)[:,1].cpu())
            else:
                for cur_video in range(batch_size):
                    pred_frames[cur_video, :] = np.array(all_outputs[:, cur_video][:,1].cpu())
            # for t in range(num_frames):
            #     # prediction
            #     # pred = all_outputs[t]['pred_mean']  # B x 2
            #     pred = all_outputs[t]
            #     pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
            #     pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)


            # gather results and ground truth
            all_pred.append(pred_frames)
            all_mil_pred.append(np.exp(mil_pred.cpu().numpy()[:, 1]) / np.sum(np.exp(mil_pred.cpu().numpy()), axis=1))
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int32)
            all_toas.append(toas)
            all_ids.append(video_ids)
            if vis:
                # gather data for visualization
                vis_data.append({'pred_frames': pred_frames, 'label': label,
                                'toa': toas, 'detections': detections, 'video_ids': video_ids})


    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    all_mil_pred = np.concatenate(all_mil_pred, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)
    print("all_mil_pred:", all_mil_pred.shape)

    return all_pred, all_labels, all_toas, all_ids, all_mil_pred





# def load_checkpoint(model, optimizer=None, filename='.', isTraining=True):
#     # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
#     start_epoch = 0
#     if os.path.isfile(filename):
#         checkpoint = torch.load(filename)
#         start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['model'])
#         print(checkpoint['args'])
#         if isTraining:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(filename))

#     return model, optimizer, start_epoch



def train_eval():
    ### --- CONFIG PATH ---
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    model_dir = os.path.join(p.output_dir, p.feature_name, p.dataset, 'snapshot', p.data_class)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # tensorboard logging
    logs_dir = os.path.join(p.output_dir, p.dataset, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # logger = SummaryWriter(logs_dir)

    # gpu options
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    print("Using GPU devices: ", gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create data loader
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    elif p.dataset == 'ours':
        from src.DataLoader import OursDataset
        data_path = '/data/yehj/SAVES/DSTA/'
        dad_data_path = '/data/yehj/SAVES/DSTA/DAD_Split'
        train_data = OursDataset(data_path, dad_data_path, p.data_class, p.feature_name, aug_type_num=p.aug_type_num, phase='train', ptm_dataset=p.ptm_dataset, box_type=p.box_type, toTensor=True, device=device, toa_modify=p.toa_modify, all50=p.all50)
        # test_data = OursDataset(data_path, p.data_class, p.feature_name, 'train', ptm_dataset=p.ptm_dataset, box_type=p.box_type, toTensor=True, device=device, vis=True)
        test_data = OursDataset(data_path, dad_data_path, p.data_class, p.feature_name, aug_type_num=1, phase='test', ptm_dataset=p.ptm_dataset, box_type=p.box_type, toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=False)
    
    # building model
    if p.model == 'IDSTA':
        model = IDSTA(train_data.dim_feature, p.hidden_dim, p.latent_dim,
                       n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps, lamda=p.lamda, tpt=p.tpt,
                       with_saa=True, with_gap=True)
    elif p.model == 'DSTA':
        model = DSTA(train_data.dim_feature, p.hidden_dim, p.latent_dim, n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps, with_saa=True)
    
    # model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)

    # optimizer
    if p.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    elif p.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=p.base_lr, momentum=p.mo, weight_decay=0.0005)

    if p.schedule == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif p.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    p.epoch,
                    eta_min=5e-6   # a tuning parameter
                    )
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.train() # set the model into training status

    # resume training
    start_epoch = -1


    iter_cur = 0
    best_metrics = {}
    best_metrics['mil_f1'] = 0
    best_metrics['mil_acc'] = 0
    best_metrics['max_f1'] = 0
    best_metrics['max_acc'] = 0

    for k in tqdm(range(p.epoch)):
        loos_per_epoch = {}
        loos_per_epoch['ce_loss'] = 0
        loos_per_epoch['aux_loss'] = 0
        loos_per_epoch['gap_loss'] = 0
        loos_per_epoch['gaploss_frame'] = 0

        loop = tqdm(enumerate(traindata_loader),total=len(traindata_loader))
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        total_loss = 0
        batch_num = 0
        lr = optimizer.param_groups[0]['lr']
        print('Cur_lr: ', lr)
        for i, (batch_xs, batch_ys, batch_toas) in loop:
            batch_num += 1
            # ipdb.set_trace()
            optimizer.zero_grad()
            losses, all_outputs, all_mil_outputs, hidden_st, alphas = model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))

            loos_per_epoch['ce_loss'] += losses['cross_entropy']
            loos_per_epoch['aux_loss'] += losses['auxloss']
            loos_per_epoch['gap_loss'] += losses['gaploss']
            loos_per_epoch['gaploss_frame'] += losses['gaploss_frame']


            losses['total_loss'] = p.loss_ce * losses['cross_entropy']
            losses['total_loss'] += p.loss_beta * losses['auxloss']
            losses['total_loss'] += p.loss_gap * losses['gaploss']
            losses['total_loss'] += p.loss_gap_frame * losses['gaploss_frame']
            total_loss += losses['total_loss']
            # print("total_loss:", losses['total_loss'])
            # backward
            if not isinstance(losses['total_loss'], float):
                losses['total_loss'].mean().backward()
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                loop.set_description(f"Epoch  [{k}/{p.epoch}]")
                loop.set_postfix(loss= losses['total_loss'].item() )
                lr = optimizer.param_groups[0]['lr']
                
                iter_cur += 1


        model.eval()
        all_pred, all_labels, all_toas, vis_data, all_mil_pred = test_all_vis(testdata_loader, model, vis=True, device=device)
        model.train()


        # save mil
        all_vid_scores = all_mil_pred
        f1_mil, threhold_mil = find_best_threshold(all_vid_scores, all_labels)
        print('Cur mil f1', f1_mil)

        all_vid_scores = [max(pred) for pred in all_pred]
        f1_max, threhold_max = find_best_threshold(all_vid_scores, all_labels)

        print('Cur max f1', f1_max)

        print('cross_entropy loss:')
        print(loos_per_epoch['ce_loss']/batch_num)
        print('auxloss loss:')
        print(loos_per_epoch['aux_loss']/batch_num)
        print('gaploss loss:')
        print(loos_per_epoch['gap_loss']/batch_num)
        print('gaploss_frame loss:')
        print(loos_per_epoch['gaploss_frame']/batch_num)
        


        # print(best_metrics)
        scheduler.step(losses['total_loss'])

    # save model
    if p.is_save:
        model_file = os.path.join(model_dir, f'{p.data_class}_best_model_mil_f1_{p.save_name}.pth')
        torch.save({'epoch': k,
                    'args': p,
                    'model': model,
                    'optimizer': optimizer.state_dict()}, model_file)
        print('Model has been saved as: %s'%(model_file))
    
    with open('./result.csv', 'a') as f:
        f.write(f'{p.save_name}, {f1_max}, {f1_mil}\n')

def find_best_threshold(all_vid_scores, all_labels):
    acc = []
    for cur_threhold in all_vid_scores:
        predict_func = lambda x: 1 if x > cur_threhold else 0
        predictions = [predict_func(i) for i in all_vid_scores]
        acc.append(sum([predictions[i] == all_labels[i] for i in range(len(all_labels))]) / len(all_labels))
    idx = np.argmax(np.array(acc))
    best_acc = max(acc)
    return best_acc, all_vid_scores[idx]

def find_best_threshold_f1(all_vid_scores, all_labels):
    from sklearn.metrics import f1_score
    f1 = []
    for cur_threhold in all_vid_scores:
        predict_func = lambda x: 1 if x > cur_threhold else 0
        predictions = [predict_func(i) for i in all_vid_scores]
        f1.append(f1_score(all_labels, predictions, average='macro'))
    idx = np.argmax(np.array(f1))
    best_f1 = max(f1)
    return best_f1, all_vid_scores[idx]

    # from sklearn.metrics import precision_recall_curve
    # precision, recall, threshold = precision_recall_curve(all_labels, all_vid_scores)
    # f1_scores = (2 * precision * recall) / (precision + recall)
    # best_f1 = np.max(f1_scores[np.isfinite(f1_scores)])
    # best_f1_idx = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # return best_f1, threshold[best_f1_idx]


def test_eval():

    # gpu options
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create data loader
    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device, vis=True)
    elif  p.dataset == 'ours':
        from src.DataLoader import OursDataset
        data_path = '/data/yehj/SAVES/DSTA/'
        dad_data_path = '/data/yehj/SAVES/DSTA/DAD_Split'
        test_data = OursDataset(data_path, dad_data_path, p.data_class, p.feature_name, aug_type_num=1, phase='test', ptm_dataset=p.ptm_dataset, box_type=p.box_type, toTensor=True, device=device, vis=True)
        # test_data = OursDataset(data_path, p.data_class, p.feature_name, 'train', toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=False)
    num_samples = len(test_data)
    print("Number of testing samples: %d"%(num_samples))

    # building model
    if p.model == 'IDSTA':
        model = IDSTA(test_data.dim_feature, p.hidden_dim, p.latent_dim,
                       n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps, lamda=p.lamda, tpt=p.tpt,
                       with_saa=True, with_gap=True)
    elif p.model == 'DSTA':
        model = DSTA(test_data.dim_feature, p.hidden_dim, p.latent_dim, n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps, with_saa=True)

    model = torch.load(p.model_file)['model']
    # model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)
    # run model inference
    all_pred, all_labels, all_toas, vis_data, all_mil_pred = test_all_vis(testdata_loader, model, vis=True, device=device)
    # save predictions


    print(p.model_file)
    all_vid_scores = all_mil_pred
    f1_mil, threhold_mil = find_best_threshold_f1(all_vid_scores, all_labels)
    acc_mil, threhold_mil = find_best_threshold(all_vid_scores, all_labels)
    print('count mil acc:', acc_mil)
    print('Cur mil f1', f1_mil)

    all_vid_scores = all_vid_scores = [max(pred) for pred in all_pred]
    f1_max, threhold_max = find_best_threshold_f1(all_vid_scores, all_labels)
    acc_max, threhold_max = find_best_threshold_f1(all_vid_scores, all_labels)
    print('count max acc:', acc_max)
    print('Cur max f1', f1_max)


    # threhold = threhold_mil
    # predict_func = lambda x: 1 if x > threhold else 0
    # predictions = [predict_func(i) for i in all_vid_scores]
    # mil_acc = sum([predictions[i] == all_labels[i] for i in range(len(all_labels))]) / len(all_labels)
    # print('count mil f1:', mil_acc)

    # #max f1
    # all_vid_scores = [max(pred) for pred in all_pred]
    # # all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
    # best_f1_max, threhold_max = find_best_threshold(all_vid_scores, all_labels)
    # print("Return max f1:", best_f1_max)
    # threhold = threhold_max
    # predict_func = lambda x: 1 if x > threhold else 0
    # predictions = [predict_func(i) for i in all_vid_scores]
    # max_acc = sum([predictions[i] == all_labels[i] for i in range(len(all_labels))]) / len(all_labels)
    # print('count max f1:', max_acc)

    with open('./result.csv', 'a') as f:
        f.write(f'{p.model_file}, {f1_max}, {f1_mil}, {acc_max}, {acc_mil}\n')

    # threhold=np.median(all_vid_scores)
    # with open(f'./{p.data_class}_test.csv', 'a') as f:
    #     f.write('film name,car accident\n')
    #     for idx, out in zip(vis_data, all_vid_scores):
    #         # cur_idx = idx['video_ids'][0]
    #         cur_label = 1 if out > threhold else 0
    #         f.write(f'{idx},{cur_label}\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='ours', choices=['dad', 'crash', 'ours'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--data_class', type=str, default='freeway', choices=['freeway', 'road'])
    parser.add_argument('--ptm_dataset', type=str, default='imagenet', choices=['imagenet', 'cars'])
    parser.add_argument('--box_type', default='center', type=str)
    parser.add_argument('--model', default='IDSTA', type=str)
    parser.add_argument('--save_name', type=str, default='best')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--mo', type=float, default=0.8)
    parser.add_argument('--toa_modify', type=int, default=10)
    parser.add_argument('--aug_type_num', type=int, default=5)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--schedule', type=str, default='Reduce')
    parser.add_argument('--tpt', type=float, default=0.06,
                        help='Soft label temperature')
    parser.add_argument('--epoch', type=int, default=30,
                        help='The number of training epoches. Default: 30')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--num_rnn', type=int, default=1,
                        help='The number of RNN cells for each timestamp. Default: 1')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='The name of feature embedding methods. Default: vgg16')
    parser.add_argument('--test_iter', type=int, default=10,
                        help='The number of iteration to perform a evaluation process. Default: 300')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='The dimension of hidden states in RNN. Default: 512')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='The dimension of latent space. Default: 256')
    parser.add_argument('--loss_beta', type=float, default=10,
                        help='The weighting factor of auxiliary loss. Default: 10')
    parser.add_argument('--loss_gap', type=float, default=10,
                        help='The weighting factor of gap loss. Default: 10')
    parser.add_argument('--loss_gap_frame', type=float, default=10)
    parser.add_argument('--loss_ce', type=float, default=1,
                        help='The weighting factor of ce loss. Default: 1')
    parser.add_argument('--lamda', type=float, default=10,
                        help='The weighting factor of pos with neg frame. Default: 10')
    parser.add_argument('--gpus', type=str, default="0",
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--phase', default='test', type=str, choices=['train', 'test'],
                        help='The state of running the model. Default: test')
    parser.add_argument('--evaluate_all', action='store_true',
                        help='Whether to evaluate models of all epoches. Default: False')
    parser.add_argument('--visualize', action='store_true',
                        help='The visualization flag. Default: False')
    parser.add_argument('--is_save', action='store_true')
    parser.add_argument('--all50', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='If to resume the training. Default: False')
    parser.add_argument('--model_file', type=str, default='./output/DSTA/vgg16/dad/snapshot_attention_v18/bayesian_gcrnn_model_23.pth',
                        help='The trained GCRNN model file for demo test only.')
    parser.add_argument('--output_dir', type=str, default='./output_debug/bayes_gcrnn/vgg16',
                        help='The directory of src need to save in the training.')
    parser.add_argument('--time_str', type=str, default='')

    p = parser.parse_args()
    print(p)
    if p.phase == 'test':
        test_eval()
    else:
        train_eval()
