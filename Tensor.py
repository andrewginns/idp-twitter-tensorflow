#!~/tensorflow
# http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html
from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import cPickle as pickle


def xprint(string):
    message = tf.constant(string)
    sess = tf.Session()
    print(sess.run(message))


def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)


"""Loading test and training data"""
trainX = pickle.load(open("unlabelled_svd_train.p", "rb"))
trainY = csv_to_numpy_array("labels_svd_train_2col.csv", delimiter=',')
xprint('\nTraining data')
xprint(trainX.shape)
xprint(trainY.shape)

testX = pickle.load(open("unlabelled_svd_test.p", "rb"))
testY = csv_to_numpy_array("labels_svd_test_2col.csv", delimiter=',')
xprint('\nTest data')
xprint(testX.shape)
xprint(testY.shape)

"""Program parameters"""

numFeatures = trainX.shape[1]
xprint(numFeatures)

# numLabels = number of authors
numLabels = trainY.shape[1]
xprint(numLabels)

"""Learning rate for optimiser"""
# numEpochs is the number of iterations
numEpochs = 200
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)

"""Temporary Tensors"""

# 'None' means no limit on number of rows
X = tf.placeholder(tf.float32, [None, numFeatures])
# yGold are the correct answers, rows have [1,0] for Trump or [0,1] for Hillary
yGold = tf.placeholder(tf.float32, [None, numLabels])

"""Variable initialisation"""

weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6 / numFeatures + numLabels + 1)),
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1, numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6 / numFeatures + numLabels + 1)),
                                    name="bias"))

"""Tensorflow operations for prediction"""

# INITIALIZE our weights and biases
init_OP = tf.global_variables_initializer()

# PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

"""Tensorflow operation for evaluation"""

# COST FUNCTION i.e. MEAN SQUARED ERROR
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")
cross_entropy = tf.reduce_mean(-tf.reduce_sum(yGold * tf.log(yGold), axis=[1]))

"""Tensorflow operation for optimisation"""

# OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
Optimizer = tf.train.GradientDescentOptimizer(learningRate)

training_OP = Optimizer.minimize(cost_OP)

"""Live updating graph"""

epoch_values = []
accuracy_values = []
cost_values = []
# Turn on interactive plotting
plt.ion()
# Create the main, super plot
fig = plt.figure()
# Create two subplots on their own axes and give titles
ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()

"""Running th program"""

# Create a tensorflow session
sess = tf.Session()

# Initialize all tensorflow variables
sess.run(init_OP)

# Ops for vizualization
# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))
# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)
# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
# Merge all summaries
all_summary_OPS = tf.summary.merge_all()
# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph_def)

# Initialize reporting variables
cost = 0
diff = 1

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence." % diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP],
                feed_dict={X: trainX, yGold: trainY}
            )
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Write summary stats to writer
            writer.add_summary(summary_results, i)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            # generate print statements
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print("step %d, cost %g" % (i, newCost))
            print("step %d, change in cost %g" % (i, diff))

            # Plot progress to our two subplots
            accuracyLine, = ax1.plot(epoch_values, accuracy_values)
            costLine, = ax2.plot(epoch_values, cost_values)
            fig.canvas.draw()
            time.sleep(1)

# How well do we perform on held-out test data?
print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                      feed_dict={X: testX,
                                                                 yGold: testY})))

"""Save trained models"""

# Create Saver
saver = tf.train.Saver()
# Save variables to .ckpt file
# saver.save(sess, "trained_variables.ckpt")

# Close tensorflow session
sess.close()
