import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn import Sequential as Seq, Linear, Tanh, ReLU
from device_util import DEVICE



class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]





class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
        
        
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)



def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).detach().numpy()
    return roc_auc_score(labels, scores)


class portHNN_split_dgl(torch.nn.Module):
    def __init__(self,g, in_dim,enc_dim, Hout,init_w=0.1):
        super(portHNN_split_dgl,self).__init__()
        self.indim = in_dim
        self.dim_split = int(in_dim/2)
        if enc_dim == in_dim:
            self.Kfun = Seq(Linear(self.dim_split, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.Ufun = Seq(Linear(self.dim_split, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.encK = Linear(self.dim_split,self.dim_split)
            self.encP1 = Linear(self.dim_split,self.dim_split)
            self.encP2 = Linear(self.dim_split,self.dim_split)
        else:
            self.Kfun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.Ufun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.encK = Linear(self.dim_split,enc_dim)
            self.encP1 = Linear(self.dim_split,enc_dim)
            self.encP2 = Linear(self.dim_split,enc_dim)
        src = g[0]
        dst = g[1]
        self.g = dgl.graph((src,dst))
        self.H = Linear(Hout, in_dim)
        self.D = Linear(in_dim,in_dim)
        nn.init.normal_(self.Kfun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[4].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[4].weight,mean=0.,std=init_w)
        
        nn.init.normal_(self.encK.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP1.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP2.weight,mean=0.,std=init_w)
        nn.init.normal_(self.H.weight,mean=0.,std=init_w)
        nn.init.normal_(self.D.weight,mean=0.,std=init_w)
    def change_graph(self,g):
        self.g = g
        
    def getK(self,x):
        p = torch.split(x, int(self.indim/2),dim=-1)[1]
        self.g.ndata["h1"] = self.encK(p)
        self.g.update_all(fn.copy_u("h1", "m"), fn.sum("m", "h1"))
        self.g.ndata["E"] = self.Kfun(self.g.ndata["h1"])
        #print(K.shape)
        return self.g.ndata["E"]
    
    def getU(self,x):
        q = torch.split(x, int(self.indim/2),dim=-1)[0]
        self.g.ndata["h2_1"] = self.encP1(q)
        self.g.ndata["h2_2"] = self.encP2(q)
        self.g.apply_edges(fn.u_add_v('h2_1', 'h2_2', 'out'))
        #print("edges out size : {}".format(self.g.edata["out"].shape))
        self.g.edata["E"] = self.Ufun(self.g.edata["out"])
        
        #print(U.shape)
        return self.g.edata["E"]
    
    def getH(self,x):
        _ = self.getK(x)
        _ = self.getU(x)
        #K = self.g.ndata["E"]
        #print("K shape {}".format(self.g.ndata["E"].shape))
        #P = self.g.edata["E"]
        #print("p shape {}".format(self.g.edata["E"].shape))
        self.g.update_all(fn.u_mul_e('E', 'E', 'm'), fn.sum('m', 'E_new'))
        return self.g.ndata["E_new"]
    
    def dHdx(self,x):
        self.getH(x)
        return self.H(self.g.ndata["E_new"])
    
    def J(self):
        M = torch.eye(self.indim)
        M = torch.cat([M[self.indim//2:], -M[:self.indim//2]]).to(DEVICE)  
        return M 
    
    def get_D(self,x):
        dh = self.dHdx(x)
        p = torch.split(dh, int(self.indim/2),dim=-1)[1]
        D_dHdx = self.D(x)
        self.g.ndata["d"] = torch.split(D_dHdx, int(self.indim/2),dim=-1)[1]
        self.g.update_all(fn.copy_u("d", "m"), fn.sum("m", "d"))
        return torch.cat((torch.zeros(p.shape).to(DEVICE),self.g.ndata["d"]),dim=-1).to(DEVICE)
    
    def forward(self,x):
        dH = self.dHdx(x)
        d = self.get_D(x)
        #no autograd it destorys learning
        #dh = torch.autograd.grad(H,x,retain_graph=True)[0]
        return dH @ self.J().transpose(0,1) - d