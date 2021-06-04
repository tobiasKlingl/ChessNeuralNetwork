"""
data_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import glob
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data(trainingsData):
    datadir="../PlayingCodeImproved/data/"
    input=datadir+trainingsData
    print("Reading in file:",input)
    datalist=glob.glob(input)

    #print("datalist=",datalist)
    training,validation,test=[],[],[]
    training_in,validation_in,test_in=[],[],[]
    training_out,validation_out,test_out=[],[],[]

    for i,input in enumerate(datalist):
        print("input",i,"=",input)
        f=open(input,'rb')
        training,validation,test= pickle.load(f)
        training_data   =training[0]
        training_out    =training[1]
        training_IDs    =training[2]
        #training_counter=training[3]
        validation_data   =validation[0]
        validation_out    =validation[1]
        validation_IDs    =validation[2]
        #validation_counter=validation[3]
        test_data   =test[0]
        test_out    =test[1]
        test_IDs    =test[2]
        #test_counter=test[3]
        print("  len(training_data)=",len(training_data))
        print("  len(training_IDs)=",len(training_IDs))
        print("  len(validation_data)=",len(validation_data))
        print("  len(validation_IDs)=",len(validation_IDs))
        
        training_in.extend(training_data)
        validation_in.extend(validation_data)
        test_in.extend(test_data)
        
        training_out.extend(training_out)
        validation_out.extend(validation_out)
        test_out.extend(test_out)
        f.close()

    #print("type(training_in))=",type(training_in[0]))
    #print("type(training_out))=",type(training_out[0]))
    #print("len(validation_in)=",len(validation_in))
    #print("len(validation_out)=",len(validation_out[0]))
    #print("len(test_in)=",len(test_in))
    #print("len(test_out)=",len(test_out))
    training_data=(training_in,training_out)
    validation_data=(validation_in,validation_out)
    test_data=(test_in,test_out)
        
    #print("training_data = ", type(training_data), type(training_data[0]), type(training_data[0][0]), type(training_data[0][1]))
    #print("training_data = ", len(training_data), len(training_data[0]), len(training_data[0][0]), len(training_data[0][1]))
    #print("len(training_data)=", len(training_data))
    #print("training_data=", training_data)
    #print("training_data[0][0] = ", training_data[0][0])
    #print("training_data[0][1] = ", training_data[0][1])
    #print("validation_data = ", len(validation_data), len(validation_data[1]))#[0]), len(validation_data[1][1])
    #print("test_data = ", len(test_data), test_data[1])#, len(test_data[1][1])

    #print training_data
    return (training_data, validation_data, test_data)

def load_data_wrapper(trainingsData):
    tr_d, va_d, te_d = load_data(trainingsData)

    inputSize=780
    training_inputs = [np.reshape(x, (inputSize, 1)) for x in tr_d[0]]
    print("training_inputs[0].shape=",training_inputs[0].shape)
    #training_results = [vectorized_result(y) for y in tr_d[1]]
    #training_data = list(zip(training_inputs, training_results))
    training_data = list(zip(training_inputs, tr_d[1]))
    #print("training_data=",training_data)
    #print("len(training_data[0][0])=",len(training_data[0][0]))
    
    validation_inputs = [np.reshape(x, (inputSize, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (inputSize, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((20, 1))
    e[j] = 1.0
    return e
