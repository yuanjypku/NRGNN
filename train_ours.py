#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN, GCNConv
from models.GCA import Encoder, GRACE, LogReg
from dataset import Dataset
import torch_geometric
from utils import load_torch_geometric_data, noisify_with_P
from models.utils import get_base_model, get_activation

# TODO: remove useless parsers and add useful
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--num_proj_hidden', type=int, default=32,
                    help='Number of hidden units of projection layer')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of layers of encoder')
parser.add_argument('--tau', type=float, default=0.4)
parser.add_argument('--dataset', type=str, default="cora", 
                    choices=['cora', 'citeseer','pubmed','dblp', 'CS', 'Computers', 'Photo'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, 
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=200, 
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, 
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.03, 
                    help='weight of loss of edge predictor')
parser.add_argument('--beta', type=float, default=1, 
                    help='weight of the loss on pseudo labels')
parser.add_argument('--t_small',type=float, default=0.1, 
                    help='threshold of eliminating the edges')
parser.add_argument('--p_u',type=float, default=0.8, 
                    help='threshold of adding pseudo labels')
parser.add_argument("--n_p", type=int, default=50, 
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50, 
                    help='number of negitive pairs per node')
parser.add_argument("--label_rate", type=float, default=0.05, 
                    help='rate of labeled data')
parser.add_argument('--noise', type=str, default='pair', choices=['uniform', 'pair'], 
                    help='type of noises')
parser.add_argument('--omega', type=float, default=0.8,
                    help='threshold of similar neighbors')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='threshold of embedding similarity, set > 1 for not using')
                    
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
np.random.seed(15) # Here the random seed is to split the train/val/test data

#%% load Dataset
if args.dataset in ['Computers', 'Photo', 'CS' ,'dblp']:
    dataset = load_torch_geometric_data('./data',args.dataset)
    adj = torch_geometric.utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])] # 只取label_rate = 5%的标注数据训练

#%% add noise to the training labels
ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise) 
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y
val_labels = labels[idx_val]
noise_y, P = noisify_with_P(val_labels,nclass, ptb, 10, args.noise) 
noise_labels[idx_val] = noise_y

# %%
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

encoder = Encoder(dataset.num_features, args.num_hidden, get_activation('prelu'), get_base_model('GCNConv'),args.num_layers)
classifier = LogReg(args.num_hidden, labels.max() + 1)
model = GRACE(args, device, encoder,classifier, args.num_hidden, args.num_proj_hidden, args.tau )
# ( param['tau']).to(device)
model.fit(features, adj, noise_labels, idx_train, idx_val)

model.logger.log("=====test set accuracy=======", _print=True)
model.test(idx_test)
model.logger.log("===================================", _print=True)
# %%
