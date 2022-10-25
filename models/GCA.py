from operator import truth
from typing import Optional

import time
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, degree
from torch_geometric.nn import GCNConv
from utils import accuracy, Logger

from models.functions import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self,args ,device, encoder,classifier, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.args = args
        self.device = device
        self.encoder = encoder
        self.classifier = classifier
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden
        self.best_val_acc = 0
        self.logger = Logger('./logs/' + args.dataset+'_l%d_p%d ' % (args.label_rate*100, 
                    args.ptb_rate*100) + time.strftime("%m-%d_%H:%M:%S", time.localtime()) )
        self.logger.log(str(args))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss_contra(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def fit(self, features, adj, labels, idx_train, idx_val, drop_scheme='degree'):
        args = self.args
        self.drop_scheme = drop_scheme

        edge_index, _ = from_scipy_sparse_matrix(adj)

        features = torch.tensor(features)
        labels = torch.tensor(labels)
        
        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.num_labels = labels.max() + 1
        self.num_nodes = edge_index.max() + 1
        self.modified_labels = self.labels.clone()

        # 计算动态增广所需的两个参数drop_weights和feature_weights TODO: 看未来需不需要加其他种类增广
        if drop_scheme== 'degree':
            drop_weights = degree_drop_weights(edge_index).to(self.device)

        if drop_scheme == 'degree':
            edge_index_ = to_undirected(edge_index)
            node_deg = degree(edge_index_[1])
            if self.args.dataset in ['Computers', 'Photo', 'CS' ,'dblp']: # dataset with onenot features
                feature_weights = feature_drop_weights(features, node_c=node_deg).to(self.device)
            elif self.args.dataset in []: # dataset with continuous features, use feature_drop_weights_dense
                raise NotImplementedError

        # Train model !
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.fc1.parameters()) +
                                        list(self.fc2.parameters()) + list(self.classifier.parameters()),
                        lr=args.lr, weight_decay=args.weight_decay)
        for epoch in tqdm(range(args.epochs)):
            self.train(epoch, idx_train, idx_val, drop_weights, feature_weights)
        self.logger.log("Optimization Finished!",_print=True)

        # Testing on val labels
        self.logger.log("picking the best model according to validation performance",_print=True)
        self.test(idx_val)

    def train(self, epoch, idx_train, idx_val, drop_weights, feature_weights):
        # 无监督网络不需要区分train 和val和test，但监督部分记着加一下
        args = self.args
        drop_scheme = self.drop_scheme
        self.encoder.train()
        self.fc1.train()
        self.fc2.train()
        self.classifier.train()

        self.optimizer.zero_grad()

        # TODO: 这里到底要不要？要是不需要对edge增广的话就删了吧
        def drop_edge(idx: int):
            return self.edge_index
            # if drop_scheme in ['degree', 'evc', 'pr']:
            #     return drop_edge_weighted(edge_index, drop_weights, p=0.3 if idx==1 else 0.4, threshold=0.7) # param[f'drop_edge_rate_{idx}']

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)

        if drop_scheme in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(self.features, feature_weights, 0.1) # param['drop_feature_rate_1']
            x_2 = drop_feature_weighted_2(self.features, feature_weights, 0.0) # param['drop_feature_rate_2']
        else:
            x_1 = drop_feature(self.features, 0.1) # param['drop_feature_rate_1']
            x_2 = drop_feature(self.features, 0.0) # param['drop_feature_rate_2'])

        z1 = self.encoder(x_1, edge_index_1)
        z2 = self.encoder(x_2, edge_index_2) # if no drop_edge, z2 is repr without mofifying graph

        loss_contra = self.loss_contra(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)

        # 用z2监督学习
        output = self.classifier(z2)
        # 对label进行修改：
        # if epoch > 20 and epoch % 5 == 0:
        if True:
            self.modify_label(z2, idx_train)
        loss_super = nn.NLLLoss()(nn.LogSoftmax()(output[idx_train]), self.modified_labels[idx_train])

        # TODO: gamma 送入args
        gamma = 4
        loss = loss_contra + gamma * loss_super

        loss.backward()
        self.optimizer.step()
        
        acc_train = accuracy(output[idx_train].detach(), self.modified_labels[idx_train])
        
        # evaluate
        self.encoder.eval()
        self.fc1.eval()
        self.fc2.eval()
        self.classifier.eval() 
        z = self.encoder(self.features, self.edge_index)
        output = self.classifier(z)
        acc_val = accuracy(output[idx_val], self.labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_encoder = deepcopy(self.encoder.state_dict())
            self.best_fc1 = deepcopy(self.fc1.state_dict())
            self.best_fc2 = deepcopy(self.fc2.state_dict())
            self.best_classifier = deepcopy(self.classifier.state_dict())

        return {'loss':loss.item(), 'acc_train': acc_train.item()}

    def test(self, idx_test):
        self.encoder.eval()
        self.fc1.eval()
        self.fc2.eval()
        self.classifier.eval() 
        self.encoder.load_state_dict(self.best_encoder)
        self.fc1.load_state_dict(self.best_fc1)
        self.fc2.load_state_dict(self.best_fc2)
        self.classifier.load_state_dict(self.best_classifier)

        z = self.encoder(self.features, self.edge_index)
        output = self.classifier(z)
        acc_test = accuracy(output[idx_test], self.labels[idx_test])

        self.logger.log("\tPredictor results:\n" +
              "accuracy= {:.4f}".format(acc_test.item()),_print=True)
        return float(acc_test)
    
    def modify_label(self, z, idx):
        # 对于每个在idx中的结点，计算它和邻居之间的cosin similarity
        # 计算每个label内满足满足cos>gamma的比例，和label内邻居的个数，只保留比例>omega的label
        # 如果不存在满足的label，不替换
        # 否则选择邻居最多的label进行替换
        with torch.no_grad():
            self.modified_labels = self.labels.clone()
            cos = nn.CosineSimilarity(dim=-1)

            x_j = self.edge_index[0]
            x_i = self.edge_index[1]

            edge_sim = cos(z[x_j],z[x_i]) > self.args.omega
            # TODO: 移动到cuda上去免得传来传去（或者看看哪个快）
            neighs_each_label = torch.zeros((self.num_labels,self.num_nodes)) + 1e-3
            simils_each_label = torch.zeros((self.num_labels,self.num_nodes))
            for e,s in enumerate(edge_sim):
                # 第一个维度是他邻居结点的label，第二个维度是他自己
                neighs_each_label[self.labels[x_j[e]]][x_i[e]] += 1
                simils_each_label[self.labels[x_j[e]]][x_i[e]] += s
            simils_each_label /= neighs_each_label
            truth_labels = simils_each_label.argmax(0)
            print('改变的个数：'+str((self.modified_labels[idx] != truth_labels[idx]).sum()))
            self.modified_labels[idx] = truth_labels[idx] #注意只改变idx的参数

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret