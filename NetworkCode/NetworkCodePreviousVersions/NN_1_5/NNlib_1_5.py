"""
Self-written neural network library Version 1.5

Use run_nn_1_5.py as an example for how to use this library

Changes (1.5):
- Performance boost by using almost exclusively numpy arrays.
"""

### Libraries
# Standard library
import json
import random
import sys
from timeit import default_timer as timer

# Third-party libraries
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p],b[p]

### Define the quadratic and cross-entropy cost functions
class QuadraticCost(object):
    @staticmethod
    def fn(a,y):
        """Return the cost associated with an output 'a' and desired output 'y'."""
        #print "Using quadratic cost"
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        """Return the error delta from the output layer."""
        return (a-y)*sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0). """
        cost=np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        return cost

    @staticmethod
    def delta(z,a,y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)

### Network class
class Network(object):
    def __init__(self, sizes, showTime = False):
        print("##########################################################")
        print("##",52*' ',"##")
        print("##  **************** NNlib Version 1.5 **************** ##")
        print("##",52*' ',"##")
        print("##########################################################")
        print("")
        print("INFO: Number of neurons per layer in the network: ",sizes)
        
        self.Time = showTime
        self.num_layers = len(sizes)
        self.sizes = sizes

    def large_weight_initializer(self):
        print("INFO: Using large_weight_initializer")
        self.biases = [ np.random.rand(y,1)/np.sqrt(y) for y in self.sizes[1:] ]# Input layer has no bias
        self.weights = [ np.random.randn(y, x)/np.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        self.bias_velocities = [ np.random.rand(y,1)/np.sqrt(y) for y in self.sizes[1:] ] # only needed for SMGD
        self.weight_velocities = [ np.random.randn(y, x)/np.sqrt(x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ] # only needed for SMGDx
        
    def default_weight_initializer(self):
        print("INFO: Using default_weight_initializer")
        self.biases = [ np.random.rand(y,1) for y in self.sizes[1:] ]
        self.weights = [ np.random.randn(y, x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]
        self.bias_velocities = [ np.random.rand(y,1) for y in self.sizes[1:] ]
        self.weight_velocities = [ np.random.randn(y, x) for y,x in zip(self.sizes[1:],self.sizes[:-1]) ]

    def feedforward(self,a,debug=False):
        a=a.transpose()
        for b,w in zip(self.biases, self.weights):
            a=sigmoid(w@a + b)
        return a

    # Stochastic gradient descent algorithm
    def SGD(self, training_data, max_epochs, n_epochs, number_eta_reductions, mini_batch_size, eta,
            lmbda = 0.0,
            regularization = "L2",
            cost = CrossEntropyCost,
            gd_technique = "SGD",
            mu = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs. Choose
        between stochastic gradient descent and stochastic-momentum-based
        gradient descent technique (``gd_technique``). ``mu`` is the 
        friction parameter (0.0 for SMGD is equivalent to using SGD).
        ``max_epochs`` is maximal number of epochs to train for and
        ``n_epochs`` is the number of epochs we keep training without
        seeing any additional improvements, before cutting the learning
        rate eta in half. ``number_eta_reductions`` is the number of
        times we cut eta in half before terminating the training.
        The other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set."""
        
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
        print("len(evaluation_data)=",len(evaluation_data))
        
        n_data = len(evaluation_data[0])
        n = len(training_data[0])

        self.evaluation_cost, self.evaluation_accuracy = [], []
        self.training_cost, self.training_accuracy = [], []

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
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                self.training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                self.training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {} ( {} % )".format(accuracy, n,100*float(accuracy)/n))
                if self.Time: print("time: {0}".format(timer() - start_timeit))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                self.evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
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
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set. """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        x = mini_batch[0].transpose()
        #if(isinstance(mini_batch[0][1],np.ndarray)):
        #   y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()
        #else:
        #   y = np.asarray([_y for _x, _y in mini_batch]).transpose()
        size=mini_batch[1].size
        y = mini_batch[1].reshape(size)
        nabla_b, nabla_w = self.backprop_matrixApproach(x,y)
        self.weights, self.biases = self.update_weights_and_biases(mu, eta, lmbda, n, size, nabla_w, nabla_b)

    def backprop_matrixApproach(self, activation, y):
        #nabla_b,nabla_w = nb.typed.List(),nb.typed.List()
        #[nabla_b.append(np.zeros(b.shape)) for b in self.biases]
        #[nabla_w.append(np.zeros(w.shape)) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        zs=[]
        activations = [activation]
        for b, w in zip(self.biases, self.weights):
            z=np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1],y)
        nabla_b[-1] = delta.sum(axis=1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_weights_and_biases(self, mu, eta, lmbda, n, m, nabla_w, nabla_b):
        if(self.GD == "SGD"): # stochastic gradient descent
            w_new = [w - eta*(lmbda/n)*Regulariation_func(self.reg,w) - (eta/m)*nw for w,nw in zip(self.weights, nabla_w)]
            b_new = [b - (eta/m)*nb for b,nb in zip(self.biases , nabla_b)]
        elif(self.GD == "SMGD"): #stochastic momentum-based gradient descent
            self.weight_velocities = [mu*v - eta*(lmbda/n)*Regulariation_func(self.reg,w) - (eta/m)*nw for v,w,nw in zip(self.weight_velocities,self.weights, nabla_w)]
            self.bias_velocities   = [mu*v - (eta/m)*nb for v,nb in zip(self.bias_velocities , nabla_b)]
            w_new = [w + v for w,v in zip(self.weights, self.weight_velocities)]
            b_new = [b + v for b,v in zip(self.biases, self.bias_velocities)]
        return w_new, b_new
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) #SYNTAX VERSTEHEN
        """
        a,y=self.feedforward(data[0]).transpose(),data[1]
        sum=0
        for ai, yi in zip(a,y):
            av,yv=ai[0],yi[0]
            diff=np.abs(av-yv)
            #if(i<500):
            #print("|av-yv|=|"+str(round(av,3))+"-"+str(yv),"=",round(diff,8))  #HIER
            if(diff<0.02):
                sum+=1
                #print(str(sum))
        return sum


    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above."""
        cost = 0.0
        x,y=data[0],data[1]
        a = self.feedforward(x,False)
        a=a.transpose()
        cost=self.cost.fn(a,y)/len(x)
        cost += 0.5*lmbda/len(x)*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
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

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    net.weight_velocities = [np.array(v) for v in data["weight_velocities"]]
    net.bias_velocities = [np.array(v) for v in data["bias_velocities"]]
    return net
        
#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def Regulariation_func(reg,w):
    func = 0.    
    if(reg == "L1"):
        func = np.sign(w)
    elif(reg == "L2"):
        func = w
    return func


