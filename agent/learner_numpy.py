import numpy as np
class MultiOutNN(object):
    def __init__(self, layer_sizes, learning_rate_init, weight_init, bias_init):
        
        self.learning_rate = float(learning_rate_init)
        self.weights, self.biases = [], []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.uniform(
                -1 * weight_init,
                weight_init,
                size=(layer_sizes[i], layer_sizes[i+1])
            ))
            self.biases.append(np.full((layer_sizes[i+1]), bias_init))
                                         
    def predict(self, observations):
        layers = self._predict(observations)
        return layers[-1]
    
    def _predict(self, observations):
        layers = [observations]
        
        for i in range(len(self.weights)):
            output = np.dot(layers[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                output = np.maximum(output, 0.01 * output)
            layers.append(output)
        
        return layers
       
    def fit(self, observations, rewards):
        layers = self._predict(observations)
        
        predictions = layers[-1]
        backprop = [np.where(np.isnan(rewards), np.zeros(rewards.shape), rewards - predictions)]
       
        for i in range(len(self.weights) - 1, 0, -1):
            layer_shape = layers[i].shape
            derivative = np.where(layers[i] >= 0, np.ones(layer_shape), np.full(layer_shape, 0.01))
            error = np.dot(backprop[-1], self.weights[i].T) * derivative
            backprop.append(error)
        backprop.reverse()
        
        error = sum(np.abs(error).sum() for error in backprop)
        if error > 1 / self.learning_rate:
            self.learning_rate /= 2.
            
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(layers[i].T, backprop[i])
            self.biases[i] += self.learning_rate * backprop[i].sum(axis=0)
