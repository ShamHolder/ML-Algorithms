import numpy as np

class Perceptron():
    def __init__(self, learningRate, nEpochs, train_X, train_y):
        # Generate random seed (currently static for testing)
        np.random.seed(1)
        
        nFeatures = X.shape[1]
        self.weights = np.array([])
        self.nEpochs = nEpochs
        self.learningRate = learningRate
        
        # Generate random weight array (is this necessary or can we start with zero weights?)
        for i in range(nFeatures+1):
            #self.weights = np.append(self.weights,2*np.random.random()-1)
            self.weights = np.append(self.weights,0.0)
            
    def predict(self,X):
        fire = self.weights[0]
        fire += np.dot(X,self.weights[1:])
        return 1.0 if fire >= 0.0 else 0.0
        
    def train(self,X,y):
        for epoch in range(self.nEpochs):
            sumError = 0.0
            for i in range(X.shape[0]):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                sumError += error**2
                self.weights[0] += self.learningRate * error
                for j in range(1,X.shape[1]+1):
                    self.weights[j] = self.weights[j] + self.learningRate * error * X[i,j-1]
            #print("Epoch = %d, learningRate = %.3f, Error = %.3f" % (epoch, self.learningRate, sumError))
