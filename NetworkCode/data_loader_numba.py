"""
data_loader (numba version)
~~~~~~~~~~~~
A library to load the played chess games data.
"""
# Standard libraries
import glob
import pickle
import gzip

# Third-party libraries
import numpy as np 
import numba as nb

def load_data(trainingsData):
    datadir="../PlayingCode_1_3/data/"
    input=datadir+trainingsData
    print("Reading in file:",input)
    datalist=glob.glob(input)

    training,validation,test=[],[],[]
    tr_in ,v_in ,te_in =[],[],[]
    tr_out,v_out,te_out=[],[],[]

    for i,input in enumerate(datalist):
        print("input",i,"=",input)
        f=open(input,'rb')
        training,validation,test= pickle.load(f)
        tr_in,tr_out,tr_IDs=np.vstack(training[0])  ,np.vstack(training[1])  ,training[2]
        v_in,v_out,v_IDs   =np.vstack(validation[0]),np.vstack(validation[1]),validation[2]
        te_in,te_out,te_IDs=test[0]                 ,test[1]                 ,test[2]
        print(" len(training_in/out/IDs)="  ,len(tr_in),"/",len(tr_out),"/",len(tr_IDs))
        print(" len(validation_in/out/IDs)=",len(v_in) ,"/",len(v_out) ,"/",len(v_IDs))
        f.close()
    return tr_in, tr_out,v_in, v_out, te_in, te_out

def load_data_wrapper(trainingsData):
    tr_in, tr_out,v_in, v_out, te_in, te_out = load_data(trainingsData)
    training_data  =nb.typed.List([tr_in,tr_out])
    validation_data=nb.typed.List([v_in,v_out])
    #if(len(te_in))>0:
    #    test_data=nb.typed.List([te_in,te_out])
    #else:
    #    test_data=nb.typed.List([te_in,te_out])#[]
    return training_data, validation_data#, test_data)
