import tensorflow as tf
import numpy as np
import csv

sess = tf.Session()


def getTrainData():
    return np.genfromtxt("spam_train.csv", delimiter=",")

def getTestData():
    return np.genfromtxt("spam_test.csv", delimiter=",")

def outputToFile(result):
    filename = "prediction2.csv"
    c = csv.writer(open(filename, "w", newline=''))
    output = []
    output.append("id")
    output.append("value")
    c.writerow(output)
    for i in range(0, len(result)):
        output = []
        output.append(str(i+1))
        output.append(sess.run(result[i])[0])
        c.writerow(output)
    
def main():    
    train_data = getTrainData()
    train_X = train_data[:, 1:train_data.shape[1]-1]
    train_Y = train_data[:, train_data.shape[1]-1]
    
    test_data = getTestData()
    test_X = test_data[:, 1:test_data.shape[1]]
    
    # data format is as usual:
    # train_X and test_X have shape (num_instances, num_features)
    # train_Y and test_Y have shape (num_instances, num_classes)
    num_features = train_X.shape[1]
    num_classes = train_Y.shape[0]
    
    # Create variables
    # X is a symbolic variable which will contain input data
    # shape [None, num_features] suggests that we don't limit the number of instances in the model
    # while the number of features is known in advance
    X = tf.placeholder("float", [num_classes, num_features])
    # same with labels: number of classes is known, while number of instances is left undefined
    Y = tf.placeholder("float",[1, num_classes])
    
    # W - weights array
    W = tf.Variable(tf.zeros([num_features, 1]))
    # B - bias array
    B = tf.Variable(tf.zeros([1]))
    
    # Define a model
    # a simple linear model y=wx+b wrapped into softmax
    pY = tf.nn.sigmoid(tf.matmul(X, W) + B)
    # pY will contain predictions the model makes, while Y contains real data
    
    # Define a cost function
    cost_fn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(pY), labels=Y))
    # You could also put it in a more explicit way
    # cost_fn = -tf.reduce_sum(Y * tf.log(pY))
    
    # Define an optimizer
    # I prefer Adam
    opt = tf.train.AdamOptimizer(0.0001).minimize(cost_fn)
    # but there is also a plain old SGD if you'd like
    #opt = tf.train.GradientDescentOptimizer(0.01).minimize(cost_fn)
    
    # Create and initialize a session
    init = tf.global_variables_initializer()
    sess.run(init)
    
    num_epochs = 4000
    for i in range(num_epochs):
		# run an optimization step with all train data
        reshape_Y = np.reshape(train_Y, [1, num_classes]);
        sess.run(opt, feed_dict={X:train_X, Y:reshape_Y})
		# thus, a symbolic variable X gets data from train_X, while Y gets data from train_Y
    
    # Now assess the model
    # create a variable which reflects how good your predictions are
    # here we just compare if the predicted label and the real label are the same
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pY,1), tf.argmax(Y,1)), "float"))
    
    tensor_test_X = tf.cast(test_X, tf.float32)
    predict_Y = tf.nn.sigmoid(tf.matmul(tensor_test_X, W) + B)
    result = []
    for i in range(0, predict_Y.shape[0]):
        result.append(tf.where(predict_Y[i] < 0.5, [0], [1]))

    outputToFile(result)
    
main()