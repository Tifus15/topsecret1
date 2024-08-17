from threebody import *
from samples import SAMPLES

alpha = torch.pi/4 # [0,pi/4]
samples =100
def transform_dataset(dataset):
    new = torch.Tensor(dataset.shape[0],dataset.shape[1],3,4)
    new[:,:,0,0:2] = dataset[:,:,0,0:2]
    new[:,:,1,0:2] = dataset[:,:,0,2:4]
    new[:,:,2,0:2] = dataset[:,:,0,4:6]
    new[:,:,0,2:4] = dataset[:,:,0,6:8]
    new[:,:,1,2:4] = dataset[:,:,0,8:10]
    new[:,:,2,2:4] = dataset[:,:,0,10:12]
    return new


maker = threebody(SAMPLES)
GOOGLES_T = round(SAMPLES["googles"]["T"],2)
YARN_T = round(SAMPLES["yarn"]["T"],2)
MOTH_T = round(SAMPLES["moth"]["T"],2)
FIG8_T = round(SAMPLES["fig8"]["T"],2)
VIII_T = round(SAMPLES["v810"]["T"],2)
print(GOOGLES_T,YARN_T,MOTH_T,FIG8_T,VIII_T)
t_googles = torch.linspace(0,GOOGLES_T-0.01,1047)
t_yarn= torch.linspace(0,YARN_T-0.01,5550)
t_moth = torch.linspace(0,MOTH_T-0.01,2867)
t_fig8 = torch.linspace(0,FIG8_T-0.01,632)
t_viii = torch.linspace(0,VIII_T-0.01,4889)
#maker.TEST(128)
"""
x,dx,t,h = maker.dataset_onekind_t("googles",samples,t_googles,[0,alpha])
x=transform_dataset(x)
dx=transform_dataset(dx)
print(x.shape)
print(dx.shape)
print(t.shape)
print(h.shape)
torch.save(x,"googles_x.pt")
torch.save(dx,"googles_dx.pt")
torch.save(h,"googles_h.pt")
torch.save(t,"googles_t.pt")

x,dx,t,h = maker.dataset_onekind_t("yarn",samples,t_yarn,[0,alpha])
x=transform_dataset(x)
dx=transform_dataset(dx)
print(x.shape)
print(dx.shape)
print(t.shape)
print(h.shape)
torch.save(x,"yarn_x.pt")
torch.save(dx,"yarn_dx.pt")
torch.save(h,"yarn_h.pt")
torch.save(t,"yarn_t.pt")

x,dx,t,h = maker.dataset_onekind_t("moth",samples,t_moth,[0,alpha])
x=transform_dataset(x)
dx=transform_dataset(dx)
print(x.shape)
print(dx.shape)
print(t.shape)
print(h.shape)
torch.save(x,"moth_x.pt")
torch.save(dx,"moth_dx.pt")
torch.save(h,"moth_h.pt")
torch.save(t,"moth_t.pt")
"""
x,dx,t,h = maker.dataset_onekind_t("fig8",samples,t_fig8,[0,alpha])
x=transform_dataset(x)
dx=transform_dataset(dx)
print(x.shape)
print(dx.shape)
print(t.shape)
print(h.shape)
torch.save(x,"fig8_x.pt")
torch.save(dx,"fig8_dx.pt")
torch.save(h,"fig8_h.pt")
torch.save(t,"fig8_t.pt")
"""
x,dx,t,h = maker.dataset_onekind_t("v810",samples,t_viii,[0,alpha])
x=transform_dataset(x)
dx=transform_dataset(dx)
print(x.shape)
print(dx.shape)
print(t.shape)
print(h.shape)
torch.save(x,"v810_x.pt")
torch.save(dx,"v810_dx.pt")
torch.save(h,"v810_h.pt")
torch.save(t,"v810_t.pt")
"""
print(t)