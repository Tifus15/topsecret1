import dgl





def create_data_Nbody(dataset,t_batchsize):
    nsnapshots = dataset.shape[0]
    nsamples = dataset.shape[1]
    #nfeats = dataset.shape[3]
    graphs=[]
    src = list(range(dataset.shape[2]))
    print("with snaphots size: {}\nwe will get {} batches from the dataset".format(t_batchsize, nsamples*int(nsnapshots-t_batchsize)))
    for i in range(int(nsnapshots-t_batchsize)):
        for j in range(int(nsamples)):
            time_range_b = i
            time_range_e = i + t_batchsize
            sample = j
            y = dataset[time_range_b:time_range_e,j,:,:]
            #print(y.shape)
            x = y[0,:,:]
            # print(x.shape)
            g = dgl.graph((src,src))
            g.ndata["x"] = x
            g.ndata["y"] = y.transpose(0,1)
            graphs.append(g)

    return graphs
    