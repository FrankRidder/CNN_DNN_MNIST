'''
DNN to classify MNIST handwritten digits
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# Parameters
# learning_rate = 0.001
# n_epochs = 20
# batch_size = 100
# display_step = 1

# Accuracy 0.9733
learning_rate = 0.001
n_epochs = 200
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
# n_hidden_3 = 64  # 1st layer number of features
# n_hidden_4 = 64  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [batch_size, n_input])
Y = tf.placeholder(tf.float32, [batch_size, n_classes])


# Create model
def custom_multilayer_perceptron(x, weights, biases):
    layers = {}
    itemNumber = 1
    layers["layer_{0}".format(itemNumber)] = tf.add(tf.matmul(x, weights.pop('h{0}'.format(itemNumber))),
                                                    biases.pop('b{0}'.format(itemNumber)))
    layers["layer_{0}".format(itemNumber)] = tf.nn.relu(layers["layer_{0}".format(itemNumber)])
    itemNumber = itemNumber + 1

    while len(weights) != 1:
        # Hidden layers
        layers["layer_{0}".format(itemNumber)] = tf.add(tf.matmul(layers["layer_{0}".format(itemNumber - 1)],
                                                                  weights.pop('h{0}'.format(itemNumber))),
                                                        biases.pop('b{0}'.format(itemNumber)))

        layers["layer_{0}".format(itemNumber)] = tf.nn.relu(layers["layer_{0}".format(itemNumber)])
        itemNumber = itemNumber + 1

    # Output layer with linear activation
    out_layer = tf.matmul(layers["layer_{0}".format(itemNumber - 1)], weights['out']) + biases['out']
    return out_layer


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    # 'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    # 'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(X, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

print("Setup Finished!")

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training 
    for i in range(n_epochs):
        total_loss = 0.
        n_batches = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            # Compute average loss
            total_loss += l
        # Display logs per epoch step
        #print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print("Optimization Finished!")

    correct_preds = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
        print(accuracy_batch)
        total_correct_preds += accuracy_batch

    print(total_correct_preds)
    print(mnist.test.num_examples)
    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
