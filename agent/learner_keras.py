from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import Constant, RandomUniform
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

class MultiOutNN(object):

    def __init__(self, layer_sizes, learning_rate_init, weight_init, bias_init, **kwargs):
        model = Sequential(**kwargs)
        kernel_initializer = RandomUniform(minval=-weight_init, maxval=weight_init)
        bias_initializer = Constant(bias_init)
        for i in range(1, len(layer_sizes)):
            model.add(Dense(
                layer_sizes[i],
                input_dim = layer_sizes[i - 1],
                kernel_initializer = kernel_initializer,
                bias_initializer = bias_initializer,
                activation = 'relu' if i < len(layer_sizes) - 1 else 'linear'
            ))

        sgd = SGD(lr=learning_rate_init)
        model.compile(loss=self.loss_function, optimizer=sgd)
        self.model = model
        self.learning_rate_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, mode='min')

    def loss_function(self, y_true, y_pred):
        diff = tf.where(tf.is_nan(y_true), tf.zeros_like(y_true), y_true - y_pred)
        return tf.square(diff)
    
    def predict(self, X):
        return self.model.predict_on_batch(X)
    
    def fit(self, X, y):
        return self.model.fit(X, y, verbose=0, batch_size=X.shape[0], epochs=1, callbacks=[self.learning_rate_scheduler])
