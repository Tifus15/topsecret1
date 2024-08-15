import torch 
import dgl
from models import *

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
#print(g)
#g = dgl.graph(([0,1,2,2,3],[1,2,2,3,4]))
#g.ndata['feat'] = torch.rand(5,10)
node_features = g.ndata['feat']
print(node_features.shape)
n_features = node_features.shape[1]
k = 5
model = Model(n_features, 100, 100)
opt = torch.optim.Adam(model.parameters())
print("node features: {}".format(node_features.shape))
print("n features: {}".format(n_features))

for epoch in range(1000):
    negative_graph = construct_negative_graph(g, k)
    #print(negative_graph)
    #print(g)
    pos_score, neg_score = model(g, negative_graph, node_features)
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

node_embeddings = model.sage(g, node_features)
print(node_embeddings.shape)


