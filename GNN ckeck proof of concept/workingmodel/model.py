import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data


class mlp(nn.Module):
    def __init__(self,acts, in_dim=2, out_dim=2, hidden=[0, 0]):
        super().__init__()
        layers = []
        layer_sizes = [in_dim] + hidden
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                       act_module(acts[layer_index-1])]
        layers += [nn.Linear(layer_sizes[-1], out_dim)]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)
        self.init_layers()
        self.config = {"act_fn": acts.__class__.__name__, "input_size": in_dim, "num_classes": out_dim, "hidden_sizes": hidden}

    def init_layers(self):
        for m in self.layers.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.,std=0.1)
                nn.init.constant_(m.bias,val=0)
    
    
    def forward(self,t, x):
       #print("foward")
        for l in self.layers:
            #print(x.shape)
            #print(l)
            x = l(x)
        #print("end")
        #print(x.shape)
        return x

class Sin(nn.Module):
    def forward(self,x):
        return torch.sin(x)
    

        
def act_module(name):
    if name == "tanh":
        return nn.Tanh()
    elif name == "sin":
        return Sin()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "relu":
        return nn.ReLU()
    elif name == "softplus":
        return nn.Softplus()
    else:
        return nn.Identity()


##############################################################

def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True,no_grad=True):
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3, 2.5))
    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index%columns]
        if no_grad:
            sns.histplot(val_dict[key], ax=key_ax, color=color, bins=50, stat=stat,
                         kde=use_kde and ((val_dict[key].max()-val_dict[key].min())>1e-8)) # Only plot kde if there is variance
            key_ax.set_title(f"{key} " + (r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape)>1 else ""))
        else:
            sns.histplot(val_dict[key].grad, ax=key_ax, color=color, bins=50, stat=stat,
                         kde=use_kde and ((val_dict[key].grad.max()-val_dict[key].grad.min())>1e-8)) # Only plot kde if there is variance
            key_ax.set_title(f"{key} " + (r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape)>1 else ""))
        
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1
    fig.subplots_adjust(wspace=0.4)
    return fig

##############################################################

def visualize_weight_distribution(model, color="C0"):
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    ## Plotting
    fig = plot_dists(weights, color=color, xlabel="Weight vals")
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()


'''
##############################################################
def visualize_weight_grad_distribution(model, color="C0",no_grad = False):
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    ## Plotting
    fig = plot_dists(weights, color=color, xlabel="Weight vals")
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()


'''
##############################################################
        
if __name__ == "__main__":
    model = mlp(["relu","sigmoid",""],7,8,[8,8,8])
    print(model.parameters())
    for name,param in model.named_parameters():
        print("{}:{}".format(name,param.data))
    for name,param in model.named_parameters():
        print("{}:{}".format(name,param.grad))
    visualize_weight_distribution(model, color="C0")
    #visualize_weight_grad_distribution(model, color="C0",no_grad=False)