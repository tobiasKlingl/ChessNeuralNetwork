import NNlib_1_7 as nn
import data_loader
from timeit import default_timer as timer
import os

start_timeit = timer()
#########################    
###  S(M)GD settings  ###
#########################
max_epochs = 500          # total number of training epochs
n_epochs = 20             # cut eta value after n_epochs without improvement
num_eta_reductions = 10
mini_batch_size = 100
eta = 0.1                  # Learning rate
Lmbda = 0.5                # regularization parameter
Mu = 0.2                   # choose between: 0.0 and 1.0 for SMGD (1.0 <-> no friction and 0.0 <-> a lot friction (equivalent to SGD))
Reg = "L1"                 # choose between: "L2" and "L1"
GD_technique = "NADAM"      # choose between: "SGD", "SMGD", "ADAM" or "NADAM"
Activation = nn.relu       # choose between: nn.sigmoid and nn.relu
#Activation = nn.sigmoid
##########################

### Load the data
iteration="0"
trainingsData="training_data_iteration_"+iteration+".pkl"
training, validation, test = data_loader.load_data_wrapper(trainingsData)

### Define and create the directory to store the network
savedir = "saves/"
iterationdir="iteration_"+iteration+"/"
if not os.path.exists(savedir+iterationdir):
    os.makedirs(savedir+iterationdir)

if  (Activation == nn.sigmoid):
    actName = "sigmoid"
    Activation_prime = nn.sigmoid_prime
elif(Activation == nn.relu):
    actName = "relu"
    Activation_prime = nn.relu_prime
else:
    print("ERROR: unknown activation function:",Activation)

### File name the network will be stored in
Outputname = savedir+iterationdir+"/NN_"+GD_technique+"_"+Reg+"_maxEpochs_"+str(max_epochs)+"_nEpochs_"+str(n_epochs)+"_numEtaRed_"+str(num_eta_reductions)+"_miniBatchSize_"+str(mini_batch_size)+"_etaInit_"+str(eta)+"_lmbda_"+str(Lmbda)+"_mu_"+str(Mu)+"_"+actName

### Define network architexture
# neurons:[input, --hidden-- ,output] layer
sizes    =[780  ,500,250,250,250,250,250,50,1]
net = nn.Network(sizes, showTime = True)

### Choose between: default_weight_initializer() and large_weight_initializer() for initializing a new network
net.large_weight_initializer()
### Alternatively, load an already existining network stored in "savedir+subdir+Inputname"
# Inputname = "NN_SMGD_L2_maxEpochs_500_nEpochs_20_numEtaRed_10_miniBatchSize_100_etaInit_1.0_lmbda_0.5_mu_0.2"
# net = nn.load(savedir+Inputname)

print("Time for initialization:",timer()-start_timeit)
net.SGD(training, max_epochs, n_epochs, num_eta_reductions, mini_batch_size, eta, lmbda = Lmbda,
        regularization = Reg,
        activationFunct = Activation,
        activationDeriv = Activation_prime,
        gd_technique = GD_technique,       
        mu = Mu, 
        evaluation_data=validation,
        monitor_evaluation = True,
        monitor_training = True)
net.save(Outputname)
print("total Time:",timer()-start_timeit)
