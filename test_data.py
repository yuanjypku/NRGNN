import numpy as np
# %% load dblp (copied from train_NRGNN.py)
from torch_geometric.datasets import CitationFull
import torch_geometric.utils as utils
dataset = CitationFull('./data','dblp')
adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
features = dataset.data.x.numpy()
labels = dataset.data.y.numpy()

# %% load Amazon Computer
from torch_geometric.datasets import Amazon, Coauthor

dA = Amazon('./data',"computers")
adjA = utils.to_scipy_sparse_matrix(dA.data.edge_index)
featuresA = dA.data.x.numpy()
labelsA = dA.data.y.numpy()

""
