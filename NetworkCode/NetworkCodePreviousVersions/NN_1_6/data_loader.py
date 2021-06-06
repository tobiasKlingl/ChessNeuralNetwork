"""
data_loader
~~~~~~~~~~~~
A library to load the played chess games data.
"""
# Standard libraries
import glob
import pickle
import gzip

# Third-party libraries
import cupy as cp   #if cupy is not available change this to "import numpy as cp"
#import numpy as cp 

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
        tr_in,tr_out,tr_IDs=cp.vstack(training[0])  ,cp.vstack(training[1])  ,training[2]
        v_in,v_out,v_IDs   =cp.vstack(validation[0]),cp.vstack(validation[1]),validation[2]
        te_in,te_out,te_IDs=test[0]                 ,test[1]                 ,test[2]
        print(" len(training_in/out/IDs)="  ,len(tr_in),"/",len(tr_out),"/",len(tr_IDs))
        print(" len(validation_in/out/IDs)=",len(v_in) ,"/",len(v_out) ,"/",len(v_IDs))
        f.close()
    return (tr_in, tr_out,v_in, v_out, te_in, te_out)

def load_data_wrapper(trainingsData):
    tr_in, tr_out,v_in, v_out, te_in, te_out = load_data(trainingsData)
    training_data  =[tr_in,tr_out]
    validation_data=[v_in,v_out]
    if(len(te_in))>0:
        test_data=[te_in,te_out]
    else:
        test_data=[]
    return (training_data, validation_data, test_data)
