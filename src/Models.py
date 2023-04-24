'''
Muhammad Monjurul Karim

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import inspect
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

'''
Note:
------------------------------------------------
#Unlike the paper this repository did not use softmax activation function in the last layer.
'''

# layers
class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=[0, 0]):
        super(AccidentPredictor, self).__init__()
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0], training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout[1], training=self.training)
        x = self.dense2(x)
        return x


#Temporal attention
class frame_AttAggregate(torch.nn.Module):
    def __init__(self, agg_dim):
        super(frame_AttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(512, 512))
        self.softmax = nn.Softmax(dim=-1)
        # initialize parameters
        import math
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens):
        """
        hiddens: (10, 512, 10) # H_(1-10)
        here, "torch" is the library
        """
        m = torch.tanh(hiddens)
        alpha = torch.softmax(torch.matmul(m,self.weight),0) # e_j
        roh = torch.mul(hiddens,alpha)
        new_h = torch.sum(roh,0)
        return new_h # (10, 512)


#Temporal Self Aggregation
class SelfAttAggregate(torch.nn.Module):
    def __init__(self, agg_dim):
        super(SelfAttAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        # initialize parameters
        import math
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens):
        """
        hiddens: (10, 19, 256, 100)
        """
        hiddens = hiddens.unsqueeze(0)
        hiddens = hiddens.permute(1,0,2,3)
        maxpool = torch.max(hiddens, dim=1)[0]
        avgpool = torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)
        # soft-attention
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1]))
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1])
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)
        return agg_feature.squeeze(dim=-1)  # (10, 512)



# class GRUNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0,0]):
#         super(GRUNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
#         self.dropout = dropout
#         # self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
#         self.dense1 = torch.nn.Linear(hidden_dim, 64)
#         self.dense2 = torch.nn.Linear(64, output_dim)
#         # self.fc = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x, h):
#         out, h = self.gru(x, h)
#         out = F.dropout(out[:,-1],self.dropout[0])
#         out = self.relu(self.dense1(out))
#         out = F.dropout(out,self.dropout[1])
#         out = self.dense2(out)
#         return out, h
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0,0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.dropout2 = nn.Dropout(p=dropout[1])
        # self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.dropout1(out[:,-1])
        # out = F.dropout(out[:,-1],self.dropout[0])
        out = self.relu(self.dense1(out))
        out = self.dropout2(out)
        # out = F.dropout(out,self.dropout[1])
        out = self.dense2(out)
        return out, h


class SpatialAttention(torch.nn.Module):
    """This is SpatialAttention."""

    def __init__(self, h_dim, z_dim, n_layers, n_obj):
        super (SpatialAttention, self).__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_obj = n_obj # 19
        self.batch_size = 10

        self.weights_att_w = Variable(torch.empty(h_dim, n_layers).normal_(mean=0.0, std = 0.5))
        self.weights_att_ua = Variable(torch.empty(h_dim, h_dim).normal_(mean=0.0, std=0.01))
        self.weights_att_ba = Variable(torch.zeros(h_dim))
        self.weights_att_wa = Variable(torch.empty(h_dim, h_dim).normal_(mean=0.0, std=0.01))

    def forward(self,obj_embed, h, t, zeros_object):
        self.weights_att_ua = self.weights_att_ua.to(h.device)
        self.weights_att_ba = self.weights_att_ba.to(h.device)
        self.weights_att_wa = self.weights_att_wa.to(h.device)
        self.weights_att_w = self.weights_att_w.to(h.device)
        brcst_w = self.weights_att_w.unsqueeze(0).repeat(self.n_obj,1,1)
        obj_embed = obj_embed.permute(1,0,2)
        image_part = torch.matmul(obj_embed, self.weights_att_ua.unsqueeze(0).repeat(self.n_obj,1,1)) + self.weights_att_ba
        h = h.permute(1,0,2)
        d = torch.matmul(h, self.weights_att_wa)
        e = torch.tanh(torch.matmul(h, self.weights_att_wa).permute(1,0,2)+ image_part)
        alphas = torch.mul(torch.softmax(torch.sum(torch.matmul(e,brcst_w),2),0),zeros_object)
#         alphas = alphas*10
        al_alphas = alphas
        obj_embed = torch.mul(alphas.unsqueeze(2),obj_embed)
        obj_embed = torch.sum(obj_embed,0)
        obj_embed = obj_embed.unsqueeze(0)
        obj_embed = obj_embed.permute(1,0,2)
        return obj_embed, al_alphas

class IDSTA(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, lamda=10, tpt=0, with_saa=True, with_gap=True):
        super(IDSTA, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.tpt = tpt
        self.lamda = lamda
        self.with_saa = with_saa
        self.with_gap = with_gap
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        #spatial_attention
        self.sp_attention = SpatialAttention(h_dim,z_dim, n_layers, n_obj)
        # rnn layer
        self.gru_net = GRUNet(h_dim+h_dim , h_dim, 2, n_layers,dropout=[0.5, 0.0])
        # self.gru_net = GRUNet(h_dim+h_dim , h_dim, 2, n_layers,dropout=[0, 0.0])
        self.frame_aggregation = frame_AttAggregate(5)
        if self.with_saa:
            # auxiliary branch
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.self_aggregation = SelfAttAggregate(self.n_frames)

        # loss function
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, x, y, toa, hidden_in=None, nbatch=80, testing=False):
        """
        :param x, (batchsize, nFrames, nBoxes, Xdim) = (10 x 100 x 20 x 4096)
        :param y, (10 x 2)
        :param toa, (10,)
        """
        losses = {'cross_entropy': 0,
                  'total_loss': 0}
        if self.with_saa:
            losses.update({'auxloss': 0})
        if self.with_gap:
            losses.update({'gaploss': 0})
            losses.update({'gaploss_frame': 0})
            
        all_outputs, all_hidden, mil_outputs = [], [], []
        all_alphas = []

        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
        else:
            h = Variable(hidden_in)
        h = h.cuda()

        alpha = torch.zeros(self.n_frames)
        zeros_object_1 = torch.sum(x[:,:,1:self.n_obj+1,:].permute(1,2,0,3),3)
        zeros_object_2 = ~zeros_object_1.eq(0)
        zeros_object = zeros_object_2.float()

        h_list = []
        # plot_soft_label = []
        # for _ in range(x.size(0)):
        #     plot_soft_label.append([])

        for t in range(x.size(1)):
            # reduce the dim of node feature (FC layer)
            #########################################################
            x_t = self.phi_x(x[:, t])
            img_embed = x_t[:, 0, :].unsqueeze(1)
            obj_embed = x_t[:, 1:, :]
            obj_embed, alphas= self.sp_attention(obj_embed, h, t, zeros_object[t])
            x_t = torch.cat([obj_embed, img_embed], dim=-1)
            h_list.append(h)
            all_alphas.append(alphas)

            if t==2:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2]),dim=0)
                h = self.frame_aggregation(h_staked)
            elif t==3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3]),dim=0)
                h = self.frame_aggregation(h_staked)
            elif t > 3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2],h_list[t-3],h_list[t-4]),dim=0)
                h = self.frame_aggregation(h_staked)

            
            output, h = self.gru_net(x_t, h)

            if self.tpt != 0:
                y_t = torch.clone(y)
                # change label for every frame 
                for cur_index, cur_y in enumerate(y[:, 1]):
                    y_t[cur_index][1] = self._soft_label((t + 1) / x.size(1), toa[cur_index] / x.size(1), margin=0.15, tpt=self.tpt) * y[cur_index][1]
                # computing losses
                L3 = self._exp_loss(output, y_t, t, toa=toa, fps=self.fps, lamda=self.lamda)
            else:
                L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
            losses['cross_entropy'] += L3
            all_outputs.append(output)
            all_hidden.append(h[-1])
            

        # import matplotlib.pyplot as plt
        # x_plot = np.arange(1, x.size(1) + 1, 1)
        # for cur_plot in range(x.size(0)):
        #     plt.plot(x_plot, plot_soft_label[cur_plot])
        #     plt.savefig(f'{cur_plot}.png')
        #     plt.clf()
        # assert 0

        if self.with_saa:
            # soft attention to aggregate hidden states of all frames
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1))
            dec = self.predictor_aux(embed_video)   # [32,2]
            mil_outputs = dec
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
            losses['auxloss'] = L4
            
        if self.with_gap:
            # 
            if torch.where(y[:,1] > 0)[0].shape[0] > 0 and torch.where(y[:,0] > 0)[0].shape[0] > 0: 
                # video loss
                samples = torch.stack(all_outputs)  #  samples: [60, batchsize, 2]
                # find all positive videos and count max frames_score
                positive_samples = torch.max(samples[:, torch.where(y[:,1] > 0)[0], 1], dim=0).values 
                negative_samples = torch.max(samples[:, torch.where(y[:,1] == 0)[0], 1], dim=0).values
                min_positive_samples = torch.min(positive_samples)
                max_negative_samples = torch.max(negative_samples)
                mean_positive_samples = torch.mean(positive_samples)
                mean_negative_samples = torch.mean(negative_samples)

                # average max(0, 1 - max(pos) + max(neg))
                L5 = (max(0, 1 - min_positive_samples + max_negative_samples)) / torch.where(y[:,1] > 0)[0].shape[0]  
                L6 = (max(0, 1 - mean_positive_samples + mean_negative_samples)) / torch.where(y[:,1] > 0)[0].shape[0]
                losses['gaploss'] = L5 + L6

                # frame loss
                accident_samples = samples[:, torch.where(y[:,1] > 0)[0], 1].permute(1,0)   # find all positive videos
                positive_outputs, negative_outputs = [], []
                L7 = 0
                margin = 3
                for cur_positive_id, cur_toa in enumerate(toa[torch.where(y[:,1] > 0)[0]].tolist()):  
                    negative_outputs = accident_samples[cur_positive_id, 0: int(cur_toa[0]) - margin]
                    positive_outputs = accident_samples[cur_positive_id, int(cur_toa[0]) + margin: ]
                    if len(negative_outputs) == 0 or len(positive_outputs) == 0:
                        print(int(cur_toa[0]))
                        continue
                    L7 += (max(0, 1 - max(positive_outputs) + max(negative_outputs))) / torch.where(y[:,1] > 0)[0].shape[0]  # average max(0, 1 - max(pos) + max(neg))
                losses['gaploss_frame'] = L7
    
        return losses, all_outputs, mil_outputs, all_hidden, all_alphas


    def _exp_loss(self, pred, target, time, toa, fps=10.0, lamda=10):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :param fps:
        :param lamda: balance between pos and neg
        :return:
        '''
        # lamda = target.shape[0] / sum(target[:,1]==1)

        # positive example (exp_loss)
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        # penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        # penalty = torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (time - toa.to(pred.dtype) - 1) / 30)
        pos_loss = -torch.mul(torch.exp(penalty).squeeze(), -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(lamda * torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss
    
    def _soft_label(self, x, toa, margin, tpt):
        '''
        y = {
            softmax(- margin / tpt) / (toa - margin) * x ,                                               0 < x <= toa - margin
            softmax[(x - toa) / tpt],                                                         toa - margin < x <  toa + margin
            [[softmax(margin / tpt) ] * (x - 60) - x + toa + margin] / (toa + margin - 60),  toa + margin <= x <= 1
        }
        '''
        x = torch.tensor(x).cuda()
        # toa = torch.tensor(toa).cuda()
        margin = torch.tensor(margin).cuda()
        tpt = torch.tensor(tpt).cuda()
        x1 = toa - margin
        x2 = toa + margin
        if x <= x1:
            y = x * torch.sigmoid(- margin / tpt) / x1
        elif x >= x2:
            y = (torch.sigmoid(margin / tpt) * (x - 1) - x + x2) / (x2 - 1)
        else:
            y = torch.sigmoid((x - toa) / tpt)
        return y

class DSTA(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True):
        super(DSTA, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps

        self.with_saa = with_saa

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        #spatial_attention
        self.sp_attention = SpatialAttention(h_dim,z_dim, n_layers, n_obj)
        # rnn layer
        self.gru_net = GRUNet(h_dim+h_dim , h_dim, 2, n_layers,dropout=[0.5, 0.0])
        # self.gru_net = GRUNet(h_dim+h_dim , h_dim, 2, n_layers,dropout=[0, 0.0])
        self.frame_aggregation = frame_AttAggregate(5)
        if self.with_saa:
            # auxiliary branch
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.self_aggregation = SelfAttAggregate(self.n_frames)

        # loss function
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, x, y, toa, hidden_in=None, nbatch=80, testing=False):
        """
        :param x, (batchsize, nFrames, nBoxes, Xdim) = (10 x 100 x 20 x 4096)
        :param y, (10 x 2)
        :param toa, (10,)
        """
        losses = {'cross_entropy': 0,
                  'total_loss': 0}
        if self.with_saa:
            losses.update({'auxloss': 0})
            losses.update({'gaploss': 0})
            losses.update({'gaploss_frame': 0})

        all_outputs, all_hidden, mil_outputs = [], [], []
        all_alphas = []

        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
        else:
            h = Variable(hidden_in)
        h = h.cuda()

        alpha = torch.zeros(self.n_frames)
        zeros_object_1 = torch.sum(x[:,:,1:self.n_obj+1,:].permute(1,2,0,3),3)
        zeros_object_2 = ~zeros_object_1.eq(0)
        zeros_object = zeros_object_2.float()

        h_list = []
        # plot_soft_label = []
        # for _ in range(x.size(0)):
        #     plot_soft_label.append([])

        for t in range(x.size(1)):
            # reduce the dim of node feature (FC layer)
            #########################################################
            x_t = self.phi_x(x[:, t])
            img_embed = x_t[:, 0, :].unsqueeze(1)
            obj_embed = x_t[:, 1:, :]
            obj_embed, alphas= self.sp_attention(obj_embed, h, t, zeros_object[t])
            x_t = torch.cat([obj_embed, img_embed], dim=-1)
            h_list.append(h)
            all_alphas.append(alphas)

            if t==2:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2]),dim=0)
                h = self.frame_aggregation(h_staked)
            elif t==3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3]),dim=0)
                h = self.frame_aggregation(h_staked)
            elif t > 3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2],h_list[t-3],h_list[t-4]),dim=0)
                h = self.frame_aggregation(h_staked)

            
            output, h = self.gru_net(x_t, h)

            # computing losses
            L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
            losses['cross_entropy'] += L3
            all_outputs.append(output)
            all_hidden.append(h[-1])
            

        # import matplotlib.pyplot as plt
        # x_plot = np.arange(1, x.size(1) + 1, 1)
        # for cur_plot in range(x.size(0)):
        #     plt.plot(x_plot, plot_soft_label[cur_plot])
        #     plt.savefig(f'{cur_plot}.png')
        #     plt.clf()
        # assert 0

        if self.with_saa:
            # soft attention to aggregate hidden states of all frames
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1))
            dec = self.predictor_aux(embed_video)   # [32,2]
            mil_outputs = dec
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
            losses['auxloss'] = L4
            

    
        return losses, all_outputs, mil_outputs, all_hidden, all_alphas


    def _exp_loss(self, pred, target, time, toa, fps=10.0):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :param fps:
        :param lamda: balance between pos and neg
        :return:
        '''
        # lamda = target.shape[0] / sum(target[:,1]==1)

        # positive example (exp_loss)
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty).squeeze(), -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss
    



if __name__ == "__main__":
    print()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # x = np.arange(0, 1, 0.01)
    # y = [soft_label(i, toa=0.6) for i in x]
    # plt.plot(x, y)
    # plt.show()
    # plt.savefig('./temp.png')
        