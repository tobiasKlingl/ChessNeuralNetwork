"""
Self-written neural network library version 1.6
Use "python run_nn.py" to use this library

Changes (1.6):
- Performance boost by using almost exclusively numpy arrays.
"""
### Standard libraries
import json
import random
import sys
from timeit import default_timer as timer

### Third-party libraries
#import cupy as cp   #if cupy is not available change this to "import numpy as cp"
import numpy as cp #Need to change this to numpy when running the playing code using multiprocessing!

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = cp.random.permutation(len(a))
    return a[p],b[p]

### Define the quadratic and cross-entropy cost functions
class QuadraticCost(object):
    @staticmethod
    def fn(a,y):
        """Return the cost associated with an output 'a' and desired output 'y'."""
        #print "Using quadratic cost"
        return 0.5*cp.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        """Return the error delta from the output layer."""
        return (a-y)*sigmoid_prime(z)


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
    def delta(z,a,y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes. """
        return (a-y)

### Network class
class Network(object):
    def __init__(self, sizes, showTime = False):
        print("##########################################################")
        print("##",52*' ',"##")
        print("##  **************** NNlib Version 1.6 **************** ##")
        print("##",52*' ',"##")
        print("##########################################################")
        print("")
        print("INFO: Number of neurons per layer in the network: ",sizes)
        
        self.Time = showTime
        self.num_layers = len(sizes)
        self.sizes = sizes

    def large_weight_initializer(self):
        print("INFO: Using large_weight_initializer")
        self.biases =[ cp.random.rand(y, 1)/cp.sqrt(y) for y   in self.sizes[1:] ]# Input layer has no bias
        self.weights=[ cp.random.randn(y,x)/cp.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        self.bias_velocities  =[ cp.random.rand(y, 1)/cp.sqrt(y) for y   in self.sizes[1:] ]                     # only needed for SMGD
        self.weight_velocities=[ cp.random.randn(y,x)/cp.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]#     -- // --
        
    def default_weight_initializer(self):
        print("INFO: Using default_weight_initializer")
        self.biases =[ cp.random.rand(y, 1) for y   in self.sizes[1:] ]
        self.weights=[ cp.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        self.bias_velocities  =[ cp.random.rand(y, 1) for y   in self.sizes[1:] ]
        self.weight_velocities=[ cp.random.randn(y,x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]

    def feedforward(self,a,debug=False):
        a=a.transpose()
        for b,w in zip(self.biases, self.weights):
            a=sigmoid(w@a + b)
        return a

    ### Stochastic gradient descent algorithm
    def SGD(self, training_data, max_epochs, n_epochs, number_eta_reductions, mini_batch_size, eta,
            lmbda = 0.0,
            regularization = "L2",
            cost = CrossEntropyCost,
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
        self.cost = cost
        self.mu = mu

        if (self.reg == "L2" or self.reg == "L1"): print("INFO: Regularization method:", self.reg)
        else:
            print("WARNING: No regularization method choosen! Won't use regularization ")
            self.reg = "none"

        if (self.GD == "SGD" or self.GD == "SMGD"):
            print("INFO: Gradient descent technique:", self.GD)
            if (self.GD == "SMGD"):
                print("INFO: friction parameter mu:", self.mu)
                if (self.mu == 0.0):
                    print("INFO: Using a friction parameter mu = 0.0 for SMGD is equivalent to using the standard stochastic gradient descent (SGD)")
            elif (self.GD == "SGD" and self.mu != 0.0):
                print("WARNING: Friction parameter mu has effect when using standard stochastic gradient descent (SGD). You might want to use SMGD." )
        else:
            print("WARNING: No Gradient descent technique choosen! Will use 'SGD'")
            self.GD = "SGD"

        print("INFO: Cost function: ",str(cost))
        n_data = len(evaluation_data[0])
        n = len(training_data[0])

        self.evaluation_cost,self.evaluation_accuracy=[],[]
        self.training_cost,  self.training_accuracy  =[],[]
        eta_initial = eta
        print("INFO: inital eta value is:",eta)
        print("--------------------------------------------------------")
        for j in range(max_epochs):
            if self.Time: start_timeit = timer()
            #if self.Time: start_shuffleTime = timer()
            training_data[0],training_data[1]=unison_shuffled_copies(training_data[0], training_data[1])
            #print("timer()-start_shuffleTime=",timer()-start_shuffleTime)
            for k in range(0,n,mini_batch_size):
                mini_batch=(training_data[0][k:k+mini_batch_size],training_data[1][k:k+mini_batch_size])
                self.update_mini_batch_matrixApproach(mini_batch, mu, eta, lmbda, n)
            if self.Time: print("Time for one training cycle (timeit):",timer()-start_timeit)
            if n_data>0 and j%1 == 0:
                print("Epoch %s training complete" % j)
            if monitor_training:
                cost,a = self.total_cost(training_data, lmbda)
                self.training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
                accuracy = self.accuracy(training_data, a)
                self.training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {} ( {} % )".format(accuracy, n,100*float(accuracy)/n))
                if self.Time: print("time: {0}".format(timer() - start_timeit))
            if monitor_evaluation:
                cost, a = self.total_cost(evaluation_data, lmbda)
                self.evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
                accuracy = self.accuracy(evaluation_data, a)
                self.evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {} ( {} % )".format(accuracy, n_data,100*float(accuracy)/n_data))
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
        self.weights,self.biases = self.update_weights_and_biases(mu, eta, lmbda, n, size, nabla_w, nabla_b)

    def backprop_matrixApproach(self, activation, y):
        nabla_b= [cp.zeros(b.shape) for b in self.biases]
        nabla_w= [cp.zeros(w.shape) for w in self.weights]
        zs=[]
        activations = [activation]
        for b, w in zip(self.biases, self.weights):
            z=cp.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1],y)
        nabla_b[-1] = delta.sum(axis=1).reshape([len(delta), 1])
        nabla_w[-1] = cp.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = cp.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = cp.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_weights_and_biases(self, mu, eta, lmbda, n, m, nabla_w, nabla_b):
        if(self.GD == "SGD"):  #stochastic gradient descent
            w_new=[w - eta*(lmbda/n)*Regularization_func(self.reg,w) - (eta/m)*nw for w,nw in zip(self.weights, nabla_w)]
            b_new=[b - (eta/m)*nb for b,nb in zip(self.biases , nabla_b)]
        elif(self.GD == "SMGD"):#stochastic momentum-based gradient descent
            self.weight_velocities = [mu*v - eta*(lmbda/n)*Regularization_func(self.reg,w) - (eta/m)*nw for v,w,nw in zip(self.weight_velocities,self.weights, nabla_w)]
            self.bias_velocities   = [mu*v - (eta/m)*nb for v,nb in zip(self.bias_velocities, nabla_b)]
            w_new=[w + v for w,v in zip(self.weights,self.weight_velocities)]
            b_new=[b + v for b,v in zip(self.biases, self.bias_velocities)]
        return w_new, b_new

    def total_cost(self, data, lmbda):
        cost = 0.0
        x,y=data[0],data[1]
        aT = self.feedforward(x,False)
        a=aT.transpose()
        cost=self.cost.fn(a,y)/len(x)
        cost+=0.5*lmbda/len(x)*sum(cp.linalg.norm(w)**2 for w in self.weights)
        return cost,a
    
    def accuracy(self, data, a):
        y=data[1]
        sum=0
        for i, (ai,yi) in enumerate(zip(a,y)):
            av,yv=ai[0],yi[0]
            diff=cp.abs(av-yv)
            #if(i<500):
            #    print("i=",i,". |av-yv|=|"+str(cp.round(av,3))+"-"+str(yv),"=",cp.round(diff,8))  #HIER
            if(diff<0.02):
                sum+=1
        return sum
    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "weight_velocities": [v.tolist() for v in self.weight_velocities],
                "bias_velocities": [v.tolist() for v in self.bias_velocities],
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
    net.weights = [cp.array(w) for w in data["weights"]]
    net.biases = [cp.array(b) for b in data["biases"]]
    net.weight_velocities = [cp.array(v) for v in data["weight_velocities"]]
    net.bias_velocities = [cp.array(v) for v in data["bias_velocities"]]
    return net
        
### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+cp.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def Regularization_func(reg,w):
    func = 0.    
    if(reg == "L1"):
        func = cp.sign(w)
    elif(reg == "L2"):
        func = w
    return func


