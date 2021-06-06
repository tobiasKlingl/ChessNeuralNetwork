"""
Self-written neural network library version 1.7
Use "python run_nn_1_7.py" to use this library

Changes (1.7):
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
import cupy as cp   #if cupy is not available change this to "import numpy as cp"
#import numpy as cp #Need to change this to numpy when running the playing code using multiprocessing!

### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+cp.exp(-z))
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    #return cp.maximum(z, 0)
    return z*(z>0)
def relu_prime(z):
    """Derivative of the relu function."""
    return 1*(z>0)

def Reg(reg,w):
    func = 0.    
    if(reg == "L1"):
        func = cp.sign(w)
    elif(reg == "L2"):
        func = w
    return func

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = cp.random.permutation(len(a))
    return a[p],b[p]

class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that cp.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*cp.log(1-a)
        returns nan.  The cp.nan_to_num ensures that that is converted
        to the correct value (0.0). """
        cost=cp.sum(cp.nan_to_num(-y*cp.log(a)-(1-y)*cp.log(1-a)))
        return cost
    @staticmethod
    def delta(a,y):
        #Return the error delta from the output layer.
        return (a-y) # Here we assume to use sigmoid for output layer!!!

    
### Network class
class Network(object):
    def __init__(self, sizes, showTime = False):
        print("##########################################################")
        print("##",52*' ',"##")
        print("##  **************** NNlib Version 1.7 **************** ##")
        print("##",52*' ',"##")
        print("##########################################################")
        print("")
        print("INFO: Number of neurons per layer in the network: ",sizes)
        
        self.Time = showTime
        self.num_layers = len(sizes)
        self.sizes = sizes

    def large_weight_initializer(self):
        print("INFO: Using large_weight_initializer")
        self.biases  =[ cp.random.rand(y, 1)/cp.sqrt(y) for y   in self.sizes[1:] ]# Input layer has no bias
        self.weights =[ cp.random.randn(y,x)/cp.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        #self.bias_v  =[ cp.random.rand(y, 1)/cp.sqrt(y) for y   in self.sizes[1:] ]                     # needed for SMGD and and Adam
        #self.weight_v=[ cp.random.randn(y,x)/cp.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]#     -- // --
        #self.bias_m  =[ cp.random.rand(y, 1)/cp.sqrt(y) for y   in self.sizes[1:] ]                     # needed for Adam
        #self.weight_m=[ cp.random.randn(y,x)/cp.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]#     -- // --
        self.bias_v  =[ cp.zeros((y, 1)) for y   in self.sizes[1:] ]                     # needed for SMGD and and Adam
        self.weight_v=[ cp.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]#     -- // --
        self.bias_m  =[ cp.zeros((y, 1)) for y   in self.sizes[1:] ]                     # needed for Adam
        self.weight_m=[ cp.zeros((y,x))  for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]#     -- // --
        
    def default_weight_initializer(self):
        print("INFO: Using default_weight_initializer")
        self.biases  =[ cp.random.rand(y, 1) for y   in self.sizes[1:] ]
        self.weights =[ cp.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        self.bias_v  =[ cp.random.rand(y, 1) for y   in self.sizes[1:] ]
        self.weight_v=[ cp.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        self.bias_m  =[ cp.random.rand(y, 1) for y   in self.sizes[1:] ]
        self.weight_m=[ cp.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]

    def feedforward(self,a,debug=False):
        a=a.transpose()
        for i,(b,w) in enumerate(zip(self.biases, self.weights)):
            if self.cost==CrossEntropyCost and i==self.num_layers-2:
                #print("using sigmoid")
                a = sigmoid(w@a + b)
            else:
                #print("using chosen activation function")
                a = self.act(w@a + b)
        return a

    ### Stochastic gradient descent algorithm
    def SGD(self, training_data, max_epochs, n_epochs, number_eta_reductions, mini_batch_size, eta,
            lmbda = 0.0,
            regularization = "L2",
            activationFunct = sigmoid,
            activationDeriv = sigmoid_prime,
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
        self.cost = CrossEntropyCost
        self.mu = mu
        self.act = activationFunct
        self.act_prime = activationDeriv
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

        print("INFO: activation:",str(activationFunct))
        print("INFO: activation (output):",str(sigmoid))
        n_data = len(evaluation_data[0])
        n = len(training_data[0])

        self.evaluation_cost,self.evaluation_accuracy=[],[]
        self.training_cost,  self.training_accuracy  =[],[]
        eta_initial = eta
        print("INFO: inital eta value is:",eta)
        print("--------------------------------------------------------")
        for j in range(max_epochs):
            if self.Time: start_timeit = timer()
            #training_data[0],training_data[1]=unison_shuffled_copies(training_data[0], training_data[1])
            for k in range(0,n,mini_batch_size):
                mini_batch=(training_data[0][k:k+mini_batch_size],training_data[1][k:k+mini_batch_size])
                self.update_mini_batch_matrixApproach(mini_batch, mu, eta, lmbda, n)
            if n_data>0 and j%1 == 0:
                print("Epoch %s training completed in %s seconds." %(j,timer()-start_timeit))
            if monitor_training and j%10==0: # Only monitor training data every 10th training epoch to speed up things
                cost,accuracy = self.cost_and_acc(training_data, lmbda)
                self.training_cost.append(cost)
                self.training_accuracy.append(accuracy)
                print("Cost on training data: {}".format(cost))
                print("Accuracy on training data: {} / {} ( {} % )".format(accuracy, n,100*accuracy/n))
                if self.Time: print("time: {0}".format(timer() - start_timeit))
            if monitor_evaluation:
                cost,accuracy = self.cost_and_acc(evaluation_data, lmbda)
                self.evaluation_cost.append(cost)
                self.evaluation_accuracy.append(accuracy)
                print("Cost on evaluation data: {}".format(cost))
                print("Accuracy on evaluation data: {} / {} ( {} % )".format(accuracy, n_data,100*accuracy/n_data))
                if self.Time: print("time: {0}".format(timer() - start_timeit))
                
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
        #nabla_b=[cp.zeros(b.shape) for b in self.biases]
        #nabla_w=[cp.zeros(w.shape) for w in self.weights]
        x = mini_batch[0].transpose()
        size=mini_batch[1].size
        y = mini_batch[1].reshape(size)
        nabla_b,nabla_w = self.backprop_matrixApproach(x,y)
        #self.weights,self.biases = self.update_weights_and_biases(mu, eta, lmbda, n, size, nabla_w, nabla_b)
        self.update_weights_and_biases(mu, eta, lmbda, n, size, nabla_w, nabla_b)

    def backprop_matrixApproach(self, activation, y):
        nabla_b= [cp.zeros(b.shape) for b in self.biases]
        nabla_w= [cp.zeros(w.shape) for w in self.weights]
        zs=[]
        activations = [activation]
        for i,(b,w) in enumerate(zip(self.biases, self.weights)):
            z=w@activation + b
            zs.append(z)
            if self.cost==CrossEntropyCost and i==self.num_layers-2:
                activation = sigmoid(z)
            else:
                activation = self.act(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(activations[-1],y)
        nabla_b[-1] = delta.sum(axis=1).reshape([len(delta), 1])
        nabla_w[-1] = delta@activations[-2].transpose()
        #nabla_w[-1] = cp.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.act_prime(z)
            delta = self.weights[-l+1].transpose()@delta * sp
            #delta = cp.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = delta@activations[-l-1].transpose()
            #nabla_w[-l] = cp.dot(delta, activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def update_weights_and_biases(self, mu, eta, lmbda, n, m, nabla_w, nabla_b):
        if  ("ADA" in self.GD):
            self.t+=1
            beta1=0.9
            beta2=0.999
            self.weight_m=[beta1*wm+(1-beta1)*nw for wm,nw in zip(self.weight_m,nabla_w)]
            self.bias_m  =[beta1*bm+(1-beta1)*nb for bm,nb in zip(self.bias_m  ,nabla_b)]
            self.weight_v=[beta2*wv+(1-beta2)*cp.power(nw,2) for wv,nw in zip(self.weight_v,nabla_w)]
            self.bias_v  =[beta2*bv+(1-beta2)*cp.power(nb,2) for bv,nb in zip(self.bias_v  ,nabla_b)]
            if  (self.GD=="ADAM"):
                wm_hat=[wm/(1-cp.power(beta1,self.t)) for wm in self.weight_m]
                bm_hat=[bm/(1-cp.power(beta1,self.t)) for bm in self.bias_m]
            elif(self.GD=="NADAM"):
                wm_hat=[(wm+(1-beta1)*nw)/(1-cp.power(beta1,self.t)) for wm,nw in zip(self.weight_m,nabla_w)]
                bm_hat=[(bm+(1-beta1)*nb)/(1-cp.power(beta1,self.t)) for bm,nb in zip(self.bias_m,nabla_b)]
            wv_hat=[wv/(1-cp.power(beta2,self.t)) for wv in self.weight_v]
            bv_hat=[bv/(1-cp.power(beta2,self.t)) for bv in self.bias_v]
            #self.weights=[w - (eta/m)*wm_h /(cp.sqrt(wv_h)+1e-8)                                 for w,wm_h,wv_h in zip(self.weights,wm_hat,wv_hat)]
            self.weights=[w - (eta/m)*wm_h /(cp.sqrt(wv_h)+1e-8) - eta*(lmbda/n)*Reg(self.reg,w) for w,wm_h,wv_h in zip(self.weights,wm_hat,wv_hat)]
            self.biases =[b - (eta/m)*bm_h /(cp.sqrt(bv_h)+1e-8)                                 for b,bm_h,bv_h in zip(self.biases ,bm_hat,bv_hat)]
        elif(self.GD == "SMGD"):#stochastic momentum-based gradient descent (variance should be called velocities in this case... )
            self.weight_v=[mu*v - (eta/m)*nw - eta*(lmbda/n)*Reg(self.reg,w) for v,w,nw in zip(self.weight_v,self.weights,nabla_w)]
            self.bias_v  =[mu*v - (eta/m)*nb                                 for v,nb   in zip(self.bias_v,nabla_b)]
            self.weights=[w+v for w,v in zip(self.weights,self.weight_v)]
            self.biases =[b+v for b,v in zip(self.biases, self.bias_v)]
        elif(self.GD == "SGD"):  #stochastic gradient descent
            self.weights=[w - eta*(lmbda/n)*Reg(self.reg,w) - (eta/m)*nw for w,nw in zip(self.weights,nabla_w)]
            self.biases =[b - (eta/m)*nb for b,nb in zip(self.biases,nabla_b)]

    def cost_and_acc(self, data, lmbda):
        cost = 0.0
        x,y=data[0],data[1]
        aT = self.feedforward(x,False)
        a=aT.transpose()
        cost=self.cost.fn(a,y)/len(x)
        cost+=0.5*lmbda/sum(w.size for w in self.weights)*sum(cp.linalg.norm(w)**2 for w in self.weights)

        diff=cp.abs(a-y)
        s=float(cp.sum(diff<0.02))
        """
        y,a=data[1].ravel().tolist(),a.ravel().tolist()
        dif=[float(abs(ai-yi)) for ai,yi in zip(a,y)]
        randmStart=random.randint(1,int(len(y))-20)
        for i in range(randmStart,randmStart+20):
            print("i= {0}. |av-yv| = |{1:.{digits}f} - {2:.{digits}f}| = {3:.{digits}f}".format(i,a[i],y[i],dif[i],digits=3))
        """
        return cost,s
        
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights" : [w.tolist() for w in self.weights],
                "biases"  : [b.tolist() for b in self.biases],
                "weight_m": [m.tolist() for m in self.weight_m],
                "bias_m"  : [m.tolist() for m in self.bias_m],
                "weight_v": [v.tolist() for v in self.weight_v],
                "bias_v"  : [v.tolist() for v in self.bias_v],
                "validation_accuracies": self.evaluation_accuracy}
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
    net = Network(data["sizes"])
    net.weights  = [cp.array(w) for w in data["weights"]]
    net.biases   = [cp.array(b) for b in data["biases"]]
    net.weight_m = [cp.array(m) for m in data["weight_m"]]
    net.bias_m   = [cp.array(m) for m in data["bias_m"]]
    net.weight_v = [cp.array(v) for v in data["weight_v"]]
    net.bias_v   = [cp.array(v) for v in data["bias_v"]]
    return net
