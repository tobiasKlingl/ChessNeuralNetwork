"""
Self-written neural network library version 1.7 (numba version)
Use "python run_nn_1_7.py" to use this library
Note: Cuda is correctly not supported for jitclass!

Changes (1.7 (numba)):
+ "Numba"-ized whole network
+ Added ReLU activation function
+ Added ADAM and NADAM optimizer
- Removed Quadrativ cost function
"""
### Standard libraries
import json
import random
import sys
from timeit import default_timer as timer

### Third-party libraries
#import cupy as cp 
import numpy as np
import numba as nb
from numba.experimental import jitclass
#from numba import cuda
#print(cuda.gpus)

@nb.extending.overload(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if isinstance(x, nb.types.Array):
        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if copy:
                out = np.copy(x).reshape(-1)
            else:
                out = x.reshape(-1)
            for i in range(len(out)):
                if np.isnan(out[i]):
                    out[i] = nan
                if posinf is not None and np.isinf(out[i]) and out[i] > 0:
                    out[i] = posinf
                if neginf is not None and np.isinf(out[i]) and out[i] < 0:
                    out[i] = neginf
            return out.reshape(x.shape)
    else:
        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if np.isnan(x):
                return nan
            if posinf is not None and np.isinf(x) and x > 0:
                return posinf
            if neginf is not None and np.isinf(x) and x < 0:
                return neginf
            return x
    return nan_to_num_impl

### Miscellaneous functions
@nb.njit(cache=True)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

@nb.njit(cache=True)
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

@nb.njit(cache=True)
def relu(z):
    #return np.maximum(z, 0)
    return z*(z>0)

@nb.njit(cache=True)
def relu_prime(z):
    """Derivative of the relu function."""
    return 1*(z>0)

@nb.njit(cache=True)
def Reg(reg,w):
    func = 0.*w 
    if(reg == "L1"):
        func = np.sign(w)
    elif(reg == "L2"):
        func = w
    return func

@nb.njit(cache=True)
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p],b[p]

@nb.njit(cache=True)
def CrossEntropyCost(a,y):
    """Return the cost associated with an output ``a`` and desired output
    ``y``.  Note that np.nan_to_num is used to ensure numerical
    stability.  In particular, if both ``a`` and ``y`` have a 1.0
    in the same slot, then the expression (1-y)*np.log(1-a)
    returns nan.  The np.nan_to_num ensures that that is converted
    to the correct value (0.0). """
    cost=np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    return cost

@nb.njit(cache=True)
def deltaCrossEntropy(a,y):
    #Return the error delta from the output layer.
    return (a-y) # Here we assume to use sigmoid for output layer!!!
    
### Network class
spec = [
    ('Time'      , nb.boolean),
    ('num_layers', nb.int64),
    ('sizes'     , nb.types.ListType(nb.types.int64)),
    ('biases'    , nb.types.ListType(nb.types.Array(nb.types.float64, 2, 'C'))),
    ('weights'   , nb.types.ListType(nb.types.Array(nb.types.float64, 2, 'C'))),
    ('bias_v'    , nb.types.ListType(nb.types.Array(nb.types.float64, 2, 'C'))),
    ('weight_v'  , nb.types.ListType(nb.types.Array(nb.types.float64, 2, 'C'))),
    ('bias_m'    , nb.types.ListType(nb.types.Array(nb.types.float64, 2, 'C'))),
    ('weight_m'  , nb.types.ListType(nb.types.Array(nb.types.float64, 2, 'C'))),
    ('GD'        , nb.types.string),
    ('reg'       , nb.types.string),
    ('cost'      , nb.float64),
    ('mu'        , nb.float64),
    ('t'         , nb.int64),
    ('evaluation_cost'    ,nb.types.ListType(nb.types.float64)),
    ('evaluation_accuracy',nb.types.ListType(nb.types.float64)),
    ('training_cost'      ,nb.types.ListType(nb.types.float64)),
    ('training_accuracy'  ,nb.types.ListType(nb.types.float64)),
]
@jitclass(spec)
class Network(object):
    def __init__(self, sizes, showTime = False):
        print("##########################################################")
        print("##",52*' ',"##")
        print("##  ******** NNlib Version 1.7 (numba version) ******** ##")
        print("##",52*' ',"##")
        print("##########################################################")
        print("")
        print("INFO: Number of neurons per layer in the network: ",sizes)
        self.Time = showTime
        self.num_layers = len(sizes)
        self.sizes = sizes

    def large_weight_initializer(self,GD):
        print("INFO: Using large_weight_initializer")
        self.biases  =nb.typed.List([ np.random.rand(y, 1)/np.sqrt(y) for y   in self.sizes[1:] ])# Input layer has no bias
        self.weights =nb.typed.List([ np.random.randn(y,x)/np.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
        if(GD=="SMGD"): # needed for SMGD
            self.bias_v  =nb.typed.List([ np.random.rand(y, 1)/np.sqrt(y) for y   in self.sizes[1:] ])                     
            self.weight_v=nb.typed.List([ np.random.randn(y,x)/np.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
            self.bias_m  =nb.typed.List([ np.zeros((y, 1)) for y   in self.sizes[1:] ])                      # <- not needed, but initialized
            self.weight_m=nb.typed.List([ np.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]) # <- not needed, but initialized
        else: # needed for Adam and Nadam
            self.bias_v  =nb.typed.List([ np.zeros((y, 1)) for y   in self.sizes[1:] ])                     
            self.weight_v=nb.typed.List([ np.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
            self.bias_m  =nb.typed.List([ np.zeros((y, 1)) for y   in self.sizes[1:] ])                     
            self.weight_m=nb.typed.List([ np.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
        
    def default_weight_initializer(self,GD):
        print("INFO: Using default_weight_initializer")
        self.biases  =nb.typed.List([ np.random.rand(y, 1) for y   in self.sizes[1:] ])
        self.weights =nb.typed.List([ np.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
        if(GD=="SMGD"):
            self.bias_v  =nb.typed.List([ np.random.rand(y, 1) for y   in self.sizes[1:] ])
            self.weight_v=nb.typed.List([ np.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
            #self.bias_m  =nb.typed.List([ np.random.rand(y, 1) for y   in self.sizes[1:] ])
            #self.weight_m=nb.typed.List([ np.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
        else:
            self.bias_v  =nb.typed.List([ np.zeros((y, 1)) for y   in self.sizes[1:] ])                     
            self.weight_v=nb.typed.List([ np.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])
            self.bias_m  =nb.typed.List([ np.zeros((y, 1)) for y   in self.sizes[1:] ])                     
            self.weight_m=nb.typed.List([ np.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ])

    def feedforward(self,a,debug=False):
        a=np.ascontiguousarray(a.transpose())
        for i,(b,w) in enumerate(zip(self.biases, self.weights)):
            if i==self.num_layers-2:
                a = sigmoid(w@a + b)
            else:
                a = relu(w@a + b)
        return a

    ### Stochastic gradient descent algorithm
    def SGD(self, training_data, max_epochs, n_epochs, number_eta_reductions, mini_batch_size, eta, lmbda = 0.0,
            regularization = "L2",
            gd_technique = "SGD",
            mu = 0.0,
            evaluation_data=None,
            monitor_evaluation=False,
            monitor_training=False):
        """Train the neural network using mini-batch stochastic gradient
        descent. Choose between stochastic gradient descent (SGD) and
        stochastic-momentum-based gradient descent (SMGd) technique 
        (``gd_technique``). ``mu`` is the friction parameter (0.0 for
        SMGD is equivalent to using SGD). ``max_epochs`` is the maximal
        number of epochs to train for and `n_epochs`` is the number of
        epochs we keep training without seeing additional improvements,
        before cutting the learning rate eta in half. 
        ``number_eta_reductions`` is the number of times we cut eta in
        half before terminating the training. The other non-optional
        parameters are self-explanatory, as is the regularization 
        parameter ``lmbda``. The method also accepts ``evaluation_data``,
        usually either the validation or test data.  We can monitor the
        cost and accuracy on either the evaluation data or the training
        data, by setting the appropriate flags."""
        self.GD = gd_technique
        self.reg = regularization
        self.mu = mu
        self.t=0
        
        if (self.reg == "L2" or self.reg == "L1"): print("INFO: Regularization method:", self.reg)
        else:
            print("WARNING: No regularization method choosen! Won't use regularization ")
            self.reg = "none"

        if (self.GD == "SGD" or self.GD == "SMGD" or "ADAM" in self.GD):
            print("INFO: Gradient descent technique:", self.GD)
            if(self.GD == "SMGD"):
                print("INFO: friction parameter mu:", self.mu)
                if (self.mu == 0.0):
                    print("INFO: Using a friction parameter mu = 0.0 for SMGD is equivalent to using the standard stochastic gradient descent (SGD)")
            elif (self.GD == "SGD" and self.mu != 0.0):
                print("WARNING: Friction parameter mu has no effect when using standard stochastic gradient descent (SGD). You might want to use SMGD." )
        else:
            print("WARNING: No Gradient descent technique choosen! Will use 'SGD'")
            self.GD = "SGD"

        #print("INFO: activation (output):",str(sigmoid))
        n_data = len(evaluation_data[0])
        n = len(training_data[0])

        #self.Players=        nb.typed.List((+1,-1))
        self.evaluation_cost=nb.typed.List([0.0])
        self.evaluation_accuracy=nb.typed.List([0.0])
        self.training_cost=nb.typed.List([0.0])
        self.training_accuracy=nb.typed.List([0.0])
        eta_initial = eta
        print("INFO: inital eta value is:",eta)
        print("--------------------------------------------------------")
        for j in range(max_epochs):
            #if self.Time: start_timeit = timer()
            training_data[0],training_data[1]=unison_shuffled_copies(training_data[0], training_data[1])
            for k in range(0,n,mini_batch_size):
                mini_batch=(training_data[0][k:k+mini_batch_size],training_data[1][k:k+mini_batch_size])
                self.update_mini_batch_matrixApproach(mini_batch, mu, eta, lmbda, n)
            if n_data>0 and j%1 == 0:
                print("Epoch",j,"training completed.")
            if monitor_training and j%10==0: # Only monitor training data every 10th training epoch to speed up things
                cost,accuracy = self.cost_and_acc(training_data, lmbda)
                if(j==0):
                    self.training_cost[0]=cost
                    self.training_accuracy[0]=accuracy
                else:
                    self.training_cost.append(cost)
                    self.training_accuracy.append(accuracy)
                print("Cost on training data:",cost)
                print("Accuracy on training data:",accuracy,"/",n,"(",100*accuracy/n,"%)")
                #if self.Time: print("time: ".timer() - start_timeit)
            if monitor_evaluation:
                cost,accuracy = self.cost_and_acc(evaluation_data, lmbda)
                if(j==0):
                    self.evaluation_cost[0]=cost
                    self.evaluation_accuracy[0]=accuracy
                else:
                    self.evaluation_cost.append(cost)
                    self.evaluation_accuracy.append(accuracy)
                print("Cost on evaluation data:",cost)
                print("Accuracy on evaluation data:",accuracy,"/",n_data,"(",100*accuracy/n_data,"%)")
                #if self.Time: print("time: ".timer() - start_timeit)
            # Early stopping
            max_eval_accuracy = max(self.evaluation_accuracy[-n_epochs:])
            if j >= n_epochs:
                print("Max. evaluation accuracy in last",n_epochs,"epochs :",100.0*max_eval_accuracy/n_data,"%")
                print("Reference evaluation accuracy on epoch",j-n_epochs,":",100.0*self.evaluation_accuracy[j-n_epochs]/n_data,"%")
                if self.evaluation_accuracy[j-n_epochs] >= max_eval_accuracy:
                    eta /= 2.0
                    if eta != eta_initial/(2.0**number_eta_reductions):
                        print("--------------------------------------------------------")
                        print("INFO: No improvement in last",n_epochs,"training epochs! Cutting eta in half!")
                        print("INFO: New eta value is",eta)
            print("--------------------------------------------------------")

            if eta == eta_initial/(2.0**number_eta_reductions):
                print("INFO: Terminating the training! Eta already cut in half",number_eta_reductions,"times.")
                break
            if j == max_epochs-1:
                print("INFO: Reached maximum number of training epochs!")
        return self.evaluation_cost, self.evaluation_accuracy, self.training_cost, self.training_accuracy
    # -------------------------------------------------------------------------------------
    
    def update_mini_batch_matrixApproach(self, mini_batch, mu, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch. """
        x = np.ascontiguousarray(mini_batch[0].transpose())
        size=mini_batch[1].size
        y = mini_batch[1].reshape(size)
        nabla_b,nabla_w = self.backprop_matrixApproach(x,y)
        #self.weights,self.biases = self.update_weights_and_biases(mu, eta, lmbda, n, size, nabla_w, nabla_b)
        self.update_weights_and_biases(mu, eta, lmbda, n, size, nabla_w, nabla_b)

    def backprop_matrixApproach(self, activation, y):
        nabla_b= nb.typed.List([np.zeros(b.shape,dtype=np.float64) for b in self.biases])
        nabla_w= nb.typed.List([np.zeros(w.shape,dtype=np.float64) for w in self.weights])
        zs=[]
        activations = [activation]
        for i,(b,w) in enumerate(zip(self.biases, self.weights)):
            z=w@activation + b
            #z=np.dot(w,np.ascontiguousarray(activation)) + b
            zs.append(z)
            if i==self.num_layers-2:
                activation = sigmoid(z)
            else:
                activation = relu(z)
            activations.append(activation)
        # backward pass
        delta = deltaCrossEntropy(activations[-1],y)
        sumDelta=delta.sum(1)
        nabla_b[-1] = sumDelta.reshape(len(delta), 1)#delta.sum(axis=1).reshape([len(delta), 1])
        nabla_w[-1] = delta@np.ascontiguousarray(activations[-2].transpose())
        #nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = self.weights[-l+1].transpose()@delta * sp
            #delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #astype(np.int64)
            sumDelta=delta.sum(1)
            nabla_b[-l] = sumDelta.reshape(len(delta),1)#delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = delta@np.ascontiguousarray(activations[-l-1].transpose())
            #nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def update_weights_and_biases(self, mu, eta, lmbda, n, m, nabla_w, nabla_b):
        if  ("ADA" in self.GD):
            self.t+=1
            beta1=0.9
            beta2=0.999
            self.weight_m=nb.typed.List([beta1*wm+(1-beta1)*nw for wm,nw in zip(self.weight_m,nabla_w)])
            self.bias_m  =nb.typed.List([beta1*bm+(1-beta1)*nb for bm,nb in zip(self.bias_m  ,nabla_b)])
            self.weight_v=nb.typed.List([beta2*wv+(1-beta2)*np.power(nw,2) for wv,nw in zip(self.weight_v,nabla_w)])
            self.bias_v  =nb.typed.List([beta2*bv+(1-beta2)*np.power(nb,2) for bv,nb in zip(self.bias_v  ,nabla_b)])
            if  (self.GD=="ADAM"):
                wm_hat=[wm/(1-np.power(beta1,self.t)) for wm in self.weight_m]
                bm_hat=[bm/(1-np.power(beta1,self.t)) for bm in self.bias_m]
            elif(self.GD=="NADAM"):
                wm_hat=[(wm+(1-beta1)*nw)/(1-np.power(beta1,self.t)) for wm,nw in zip(self.weight_m,nabla_w)]
                bm_hat=[(bm+(1-beta1)*nb)/(1-np.power(beta1,self.t)) for bm,nb in zip(self.bias_m,nabla_b)]
            wv_hat=[wv/(1-np.power(beta2,self.t)) for wv in self.weight_v]
            bv_hat=[bv/(1-np.power(beta2,self.t)) for bv in self.bias_v]
            self.weights=nb.typed.List([w - (eta/m)*wm_h /(np.sqrt(wv_h)+1e-8) - eta*(lmbda/n)*Reg(self.reg,w) for w,wm_h,wv_h in zip(self.weights,wm_hat,wv_hat)])
            self.biases =nb.typed.List([b - (eta/m)*bm_h /(np.sqrt(bv_h)+1e-8)                                 for b,bm_h,bv_h in zip(self.biases ,bm_hat,bv_hat)])
        elif(self.GD == "SMGD"):#stochastic momentum-based gradient descent (variance should be called velocities in this case... )
            self.weight_v=nb.typed.List([mu*v - (eta/m)*nw - eta*(lmbda/n)*Reg(self.reg,w) for v,w,nw in zip(self.weight_v,self.weights,nabla_w)])
            self.bias_v  =nb.typed.List([mu*v - (eta/m)*nb                                 for v,nb   in zip(self.bias_v,nabla_b)])
            self.weights=nb.typed.List([w+v for w,v in zip(self.weights,self.weight_v)])
            self.biases =nb.typed.List([b+v for b,v in zip(self.biases, self.bias_v)])
        elif(self.GD == "SGD"):  #stochastic gradient descent
            self.weights=nb.typed.List([w - eta*(lmbda/n)*Reg(self.reg,w) - (eta/m)*nw for w,nw in zip(self.weights,nabla_w)])
            self.biases =nb.typed.List([b - (eta/m)*nb for b,nb in zip(self.biases,nabla_b)])

    def cost_and_acc(self, data, lmbda):
        cost = 0.0
        x,y=data[0],data[1]
        aT = self.feedforward(x,False)
        a=aT.transpose()

        cost=CrossEntropyCost(a,y)/len(x)
        #cost+=0.5*lmbda/sum(w.size for w in self.weights)*sum(np.linalg.norm(w)**2 for w in self.weights)
        sumSize,sumNorm=0.,0.
        for w in self.weights:
            sumSize+=w.size
            sumNorm+=np.linalg.norm(w)**2
        cost+=0.5*lmbda/sumSize*sumNorm

        diff=np.abs(a-y)
        s=float(np.sum(diff<0.02))

        y,a=data[1].ravel(),a.ravel()
        dif=nb.typed.List([float(abs(ai-yi)) for ai,yi in zip(a,y)])
        randmStart=random.randint(1,int(len(y))-20)
        for i in range(randmStart,randmStart+20):
            print("i=",i,",|av-yv|=|",np.round(a[i],3),"-",np.round(y[i],3),"| =",np.round(dif[i],3))
        return cost,s


def convert_to_python(sizes,weights,biases,weight_m,bias_m,weight_v,bias_v,evaluation_accuracy):
    Sizes    =[i for i in sizes]
    Weights  =[w.tolist() for w in weights]
    Biases   =[b.tolist() for b in biases]
    Weights_m=[m.tolist() for m in weight_m]
    Biases_m =[m.tolist() for m in bias_m]
    Weights_v=[v.tolist() for v in weight_v]
    Biases_v =[v.tolist() for v in bias_v]
    EvalAcc  =[i for i in evaluation_accuracy]
    return Sizes,Weights,Biases,Weights_m,Biases_m,Weights_v,Biases_v,EvalAcc

def save(sizes,weights,biases,weight_m,bias_m,weight_v,bias_v,evaluation_accuracy, filename):
    """Save the neural network to the file ``filename``."""
    data = {"sizes": sizes,
            "weights" : [w for w in weights],
            "biases"  : [b for b in biases],
            "weight_m": [m for m in weight_m],
            "bias_m"  : [m for m in bias_m],
            "weight_v": [v for v in weight_v],
            "bias_v"  : [v for v in bias_v],
            "validation_accuracies": evaluation_accuracy}
    f = open(filename, "w")
    json.dump(data, f)
    f.close()

### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    Sizes        = data["sizes"]
    net          = Network(nb.typed.List(Sizes),showTime=True)
    net.weights  = nb.typed.List([np.array(w) for w in data["weights"]])
    net.biases   = nb.typed.List([np.array(b) for b in data["biases"]])
    net.weight_m = nb.typed.List([np.array(m) for m in data["weight_m"]])
    net.bias_m   = nb.typed.List([np.array(m) for m in data["bias_m"]])
    net.weight_v = nb.typed.List([np.array(v) for v in data["weight_v"]])
    net.bias_v   = nb.typed.List([np.array(v) for v in data["bias_v"]])
    return net
