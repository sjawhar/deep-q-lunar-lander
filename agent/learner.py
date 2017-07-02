import tensorflow as tf
class MultiOutNN:
    
    def __init__(self, layer_sizes, learning_rate_init):
        self.learning_rate_init = learning_rate_init
        self.layer_sizes = layer_sizes

        self.observations = tf.placeholder(tf.float32, shape=[None, layer_sizes[0]])
        self.actions = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.learning_rate = tf.placeholder(tf.float32)
        self.session = tf.Session()
        
        stddev = 0.1
        mean = 0
        bias = -0.001
        
        layers = [self.observations]
        weights, biases = [], []
        
        for i in range(len(layer_sizes) - 1):
            weights.append(tf.Variable(tf.truncated_normal(
                [layer_sizes[i], layer_sizes[i+1]], 
                mean=mean,
                stddev=stddev
            )))
            biases.append(tf.Variable(tf.constant(
                bias,
                shape=[layer_sizes[i+1]]
            )))

        for i in range(len(layer_sizes) - 2):
            layers.append(self.activation_function(
                tf.matmul(layers[i], weights[i]) + biases[i]
            ))
        
        self.predictions = tf.matmul(layers[-1], weights[-1]) + biases[-1]
        action_predictions = tf.reduce_sum(tf.multiply(self.predictions, self.actions), axis=1)
        loss = tf.reduce_sum(tf.square(tf.subtract(self.rewards, action_predictions)))
        
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)
    
    def activation_function(self, input_values):
        return tf.maximum(0.01 * input_values, input_values)
            
    def predict(self, observations):
        return self.session.run(
            self.predictions,
            feed_dict = {self.observations: observations}
        )
       
    def fit(self, observations, actions, rewards):                      
        self.session.run(
            self.train,
            feed_dict = {
                self.observations: observations,
                self.actions: actions,
                self.rewards: rewards,
                self.learning_rate: self.learning_rate_init
            }
        )
