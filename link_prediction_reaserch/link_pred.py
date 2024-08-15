import dgl
from device_util import ROOT_PATH, DEVICE
import torch
from datasets import makeDGL3dofDataset
import scipy.sparse as sp
import numpy as np
from link_models import *
import itertools

filename = "trajdof3.pt"


edges = [[0,1,1,2],[1,2,0,1]]


dataset_base = torch.load(ROOT_PATH + "/"+ filename)

data = dataset_base[:,0,:,:]

cutted_data = makeDGL3dofDataset(edges,data.unsqueeze(1),2)
g = cutted_data[0]

u, v = g.edges()

eids = np.arange(g.num_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.2)
train_size = g.num_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0)
print(neg_u, neg_v)

neg_eids = np.random.choice(len(neg_u), g.num_edges())
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]],
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]],
)
train_g = dgl.remove_edges(g, eids[:test_size])

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())



model = portHNN_split_dgl(edges,2,100,100)
model.change_graph(train_g)
pred = MLPPredictor(2)


optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.01
)

all_logits = []
for e in range(1000):
    # forward
    model.change_graph(train_g)
    h = model(train_g.ndata["x"])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))
        

