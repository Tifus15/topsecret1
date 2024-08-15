from real_sim import *
import dgl
import matplotlib.pyplot as plt

def make_datasets(t,nodes=5,batches=100,method="dopri5"):
    src0 = src_list(nodes-1)
    dst0 = dst_list(nodes-1)
    N=nodes
    g0=dgl.graph((src0,dst0))
    g0.ndata["m"] = torch.ones(N-1,1)
    src1 = src_list(nodes)
    dst1 = dst_list(nodes)
    g1=dgl.graph((src1,dst1))
    g1.ndata["m"] = torch.ones(N,1)
    
    x_first=[]
    x_second=[]
    h_first=[]
    h_second=[]
    dx_first=[]
    dx_second=[]
    acc=0
    while(acc<batches):
        print(acc)
        model =G_Nbody(g0,1.00,1e-5)
        solver = BODY_solver(model)
        x0 = torch.rand(N-1,6)
        x = solver(t,x0,method)
        H = solver.H(x)
        dx = solver.dx(x)
        print(H)
        H_0 = torch.abs(H -torch.mean(H))
        H_max = torch.max(H_0)
        H_mean = torch.abs(torch.mean(H))
        crit =H_max/H_mean
        print(crit)
        if crit > 0.05:
            continue
        else:
            flag=False
            model =G_Nbody(g1,1.00,1e-4)
            solver = BODY_solver(model)
            while(not flag):
                newx0 = torch.cat((x0,torch.rand(1,6)),dim=0)
                newx = solver(t,newx0,method)
                newH = solver.H(newx)
                print(newH)
                newdx = solver.dx(newx) 
                H_0 = torch.abs(newH -torch.mean(newH))
                H_max = torch.max(H_0)
                H_mean = torch.abs(torch.mean(newH))
                crit =H_max/H_mean
                print(crit)
                if crit <= 0.05:
                    flag=True
            x_first.append(x.unsqueeze(1))
            x_second.append(newx.unsqueeze(1))
            h_first.append(H.unsqueeze(1))
            h_second.append(newH.unsqueeze(1))
            dx_first.append(dx.unsqueeze(1))
            dx_second.append(newdx.unsqueeze(1))
            print("appended")
        acc+=1
    data0x = torch.cat((x_first),dim=1)
    data1x = torch.cat((x_second),dim=1)
    data0dx = torch.cat((dx_first),dim=1)
    data1dx = torch.cat((dx_second),dim=1)
    data0H = torch.cat((h_first),dim=1)
    data1H = torch.cat((h_second),dim=1)
    torch.save(data0x,"{}_x.pt".format(N-1))
    torch.save(data0dx,"{}_dx.pt".format(N-1))
    torch.save(data0H,"{}_h.pt".format(N-1))
    torch.save(data1x,"{}_x.pt".format(N))
    torch.save(data1dx,"{}_dx.pt".format(N))
    torch.save(data1H,"{}_h.pt".format(N))
    return [data0x,data0dx,data0H], [data1x,data1dx,data1H]

#t = torch.linspace(0,1.27,128)

#data4, data5 = make_datasets(t,batches = 500)
            







N=10
R=10
dst = dst_list(N)
src = src_list(N)
g = dgl.graph((src,dst))
g.ndata["m"] = torch.ones(N,1)
pos = -R + torch.rand(N , 2)* 2*R
vel = -R + torch.rand(N , 2)*2*R 
#vel -= torch.mean(vel,dim=0)
x0 = torch.cat((pos,vel),dim=-1)
"""
extra_pos = torch.rand(1,3)
extra_vel = torch.zeros(1,3)
extra = torch.cat((extra_pos,extra_vel),dim=1)
x0 = torch.cat((x0,extra),dim=0)
dst = dst_list(N+1)
src = src_list(N+1)
g = dgl.graph((src,dst))
g.ndata["m"] = torch.ones(N+1,1)
"""
model =G_Nbody(g,1.0,1e-3)
solver = BODY_solver(model)

t = torch.linspace(0,1.27,128)
#print(t)

x = solver(t,x0,"rk4")
H, K, P = solver.H(x)
plt.figure()
plt.plot(t,H.detach().numpy())
plt.plot(t,K.detach().numpy())
plt.plot(t,P.detach().numpy())
plt.legend(["H","K","P"])
plt.show()
#print(x)
#print(H)
H_mean = torch.mean(H)
H_std = torch.std(H)
#print(H_mean)
#print(H_std)
print(torch.mean(H-H_mean))
print(H)


if x0.shape[1]==4:
    fig = plt.figure()
    for i in range(len(t)):
        plt.cla()
        for j in range(x.shape[1]):
            point = x[i,j,0:3]
            plt.scatter(point[0],point[1])
            plt.xlim(-2*R,2*R)
            plt.ylim(-2*R,2*R)
        
        plt.pause(0.005)
else:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(t)):
        plt.cla()
        for j in range(x.shape[1]):
            point = x[i,j,0:3]
            ax.scatter(point[0],point[1],point[2])
            ax.set_xlim(-R,R)
            ax.set_ylim(-R,R)
            ax.set_zlim(-R,R)
        plt.pause(0.005)
   
