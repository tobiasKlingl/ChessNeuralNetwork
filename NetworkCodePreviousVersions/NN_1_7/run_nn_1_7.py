from timeit import default_timer as timer
import os

version="cupy"
#version="numba"

if   version=="cupy" :
    import NNlib_1_7_cupy as nn
    import data_loader_cupy as data_loader
elif version=="numba":
    import NNlib_1_7_numba as nn
    import data_loader_numba as data_loader
    import numba as nb

start_timeit = timer()
##################    
###  settings  ###
##################
maxEpochs = 500      # total number of training epochs
nEpochs = 20         # cut eta value after nEpochs without improvement
EtaRed = 20          # number of eta reductions
mBatchSize = 100     # Size of mini batches
eta = 0.1            # Learning rate
Lmbda = 0.5          # regularization parameter
Mu = 0.2             # choose between: 0.0 and 1.0 for SMGD (1.0 <-> no friction and 0.0 <-> a lot friction (equivalent to SGD))
Reg = "L2"           # choose between: "L2" and "L1"
GD = "NADAM"         # choose between: "SGD", "SMGD", "ADAM" or "NADAM"
Activation = nn.relu # choose between: nn.sigmoid and nn.relu  (not supported for numba!)
##########################

### Load the data
iteration="0"
trainingsData="training_data_iteration_"+iteration+".pkl"
training, validation = data_loader.load_data_wrapper(trainingsData)

### Define and create the directory where the network is stored
Dir="saves/iteration_"+iteration+"/"
if not os.path.exists(Dir):
    os.makedirs(Dir)

if  (Activation == nn.sigmoid): # only needed for cupy version!
    act = "sigmoid"
    Activation_prime = nn.sigmoid_prime
elif(Activation == nn.relu):
    act = "relu"
    Activation_prime = nn.relu_prime
else:
    print("ERROR: unknown activation function!")

### Define network architexture
if   version=="cupy":
    sizes =[780  ,500,250,50,1]
elif version=="numba":
    sizes =nb.typed.List((780,500,250,50,1))

### Initialize a new network
net = nn.Network(sizes, showTime = True)
net.large_weight_initializer(GD) # default_weight_initializer or large_weight_initializer() 
### Load an already existining network (stored in "Dir+Inputname")
# Inputname = "NN_SMGD_L2_maxEpochs_500_nEpochs_20_EtaRed_10_mBatchSize_100_etaInit_1.0_lmbda_0.5_mu_0.2"
# net = nn.load(Dir+Inputname)

print("Time for initialization:",timer()-start_timeit)

### Now train the network
if version=="cupy":
    Outputname = Dir+GD+"_"+Reg+"_maxEpochs_"+str(maxEpochs)+"_nEpochs_"+str(nEpochs)+"_EtaRed_"+str(EtaRed)+"_mBatchSize_"+str(mBatchSize)+"_eta_"+str(eta)+"_lmbda_"+str(Lmbda)+"_mu_"+str(Mu)+"_"+act
    net.SGD(training,maxEpochs,nEpochs,EtaRed,mBatchSize,eta,Lmbda,Reg,Activation,Activation_prime,GD,Mu,validation,True,True)
    net.save(Outputname)
elif version=="numba":
    Outputname = Dir+GD+"_"+Reg+"_maxEpochs_"+str(maxEpochs)+"_nEpochs_"+str(nEpochs)+"_EtaRed_"+str(EtaRed)+"_mBatchSize_"+str(mBatchSize)+"_eta_"+str(eta)+"_lmbda_"+str(Lmbda)+"_mu_"+str(Mu)+"_relu"
    net.SGD(training,maxEpochs,nEpochs,EtaRed,mBatchSize,eta,Lmbda,Reg,GD,Mu,validation,True,True)
    Sizes,Weights,Biases,Weights_m,Biases_m,Weights_v,Biases_v,EvalAcc=nn.convert_to_python(net.sizes,net.weights,net.biases,net.weight_m,net.bias_m,net.weight_v,net.bias_v,net.evaluation_accuracy)
    nn.save(Sizes,Weights,Biases,Weights_m,Biases_m,Weights_v,Biases_v,EvalAcc, Outputname)

print("total Time:",timer()-start_timeit)
