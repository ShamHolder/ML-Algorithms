import numpy as np

class LogisticRegression():
    def __init__(self, numLayers=2, layerSizes=np.array([3,32,32,1]), alpha=1):
        # Generate random seed (currently static for testing)
        np.random.seed(1)
        
        self.synapticWeights = []
        self.numLayers = numLayers
        self.alpha = alpha
        
        # Generate a matrix to store our synaptic weights
        for layer in range(numLayers+1):
            self.synapticWeights.append(2 * np.random.random((layerSizes[layer],layerSizes[layer+1])) - 1)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def train(self, X, y, iterations):
        
        #print("Initial Weights:")
        #print(self.synapticWeights[0])
        #print(self.synapticWeights[1])
        
        alpha = self.alpha
        print("Training for Alpha: " + str(alpha))
        
        # begin iteration loop
        for i in range(iterations):
            
            nextInput = X
    
            #print(X)
            #print(self.synapticWeights[0])
            #print(np.dot(X,self.synapticWeights[0]))
            
            if self.numLayers == 0:
                output = self.think(X,0)
                error = y - output
                self.synapticWeights[0] += alpha*(np.dot(X.T, error * self.sigmoid_deriv(output)))
            
            else:
                layers = []
                deltas = []
                # feed-forward through all layers
                for layer in range(self.numLayers+1):
                    layers.append(self.think(nextInput, layer))
                    nextInput = layers[layer]
                    
                #print("Layers:")
                #print(layers[0])
                #print(layers[self.numLayers])
              
                #Calculate error on output layer
                deltas.append((y - layers[self.numLayers])*self.sigmoid_deriv(layers[self.numLayers]))
                
                if i % 10000 == 0:
                    print("Error after "+str(i)+" iterations:" + str(np.mean(np.abs(y - layers[self.numLayers]))))
            
                # Back-propagate deltas
                for layer in range(self.numLayers,0,-1):
                    deltas.append(deltas[layer-self.numLayers].dot(self.synapticWeights[layer].T)*self.sigmoid_deriv(layers[layer-1]))
            
                #print(deltas[0])
                #print(deltas[1])
                #print(X.T.dot(deltas[self.numLayers-0]))
                #print(layers[self.numLayers-1])
                #print(deltas[self.numLayers-1])
                #print(layers[self.numLayers-1].T.dot(deltas[self.numLayers-1]))
            
                # Update synaptic weights
                for layer in range(self.numLayers,-1,-1):
                    if layer == 0:
                        self.synapticWeights[layer] += alpha * (X.T.dot(deltas[self.numLayers-layer]))
                    else:
                        self.synapticWeights[layer] += alpha * (layers[self.numLayers-layer].T.dot(deltas[self.numLayers-layer]))
                
                #print("Updated Weights")
                #print(self.synapticWeights[0])
                #print(self.synapticWeights[1])
    
    def think(self, inputs, layer):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synapticWeights[layer]))
        return output
    
    def predict(self, inputs):
        inputs = inputs.astype(float)
        for layer in range(self.numLayers+1):
            inputs = self.sigmoid(np.dot(inputs, self.synapticWeights[layer]))
            
        return inputs
