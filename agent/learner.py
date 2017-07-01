import tensorflow as tf
class CustomMultiOutNN:
    
    def __init__(self, layer_sizes, learning_rate_init):
        self.learning_rate_init = learning_rate_init
        self.layer_sizes = layer_sizes
        
        std_dev = 0.1
        mean = 0.00
        bias = -0.001

        self.observations = tf.placeholder(tf.float32, shape=[None, layer_sizes[0]])
        self.actions = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.learning_rate = tf.placeholder(tf.float32)
        
        layers = [self.observations]
        weights, biases = [], []
        
        for i in range(len(layer_sizes) - 1):
            weights.append(tf.Variable(tf.truncated_normal(
                [layer_sizes[i], layer_sizes[i+1]], 
                mean=mean,
                stddev=std_dev
            )))
            biases.append(tf.Variable(tf.constant(
                bias,
                shape=[layer_sizes[i+1]]
            )))

        for i in range(len(layer_sizes) - 2):
            layers.append(self.activation_function(
                tf.matmul(layers[i], weights[i]) + biases[i]
            ))
        
        predictions = tf.matmul(layers[-1], weights[-1]) + biases[-1]
        action_predictions = tf.reduce_sum(tf.multiply(predictions, self.actions), axis=1)
        loss = tf.reduce_sum(tf.square(self.rewards - action_predictions))
        
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.session = tf.Session()

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)
    
    def activation_function(self, input_value):
        return tf.maximum(0.01 * input_value, input_value)
            
    def predict(self, observations):
        return self.session.run(
            self.rewards,
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