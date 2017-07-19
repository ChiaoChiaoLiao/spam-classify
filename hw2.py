# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:06:28 2017

@author: Chuchiao_Liao
"""
import numpy as np
import csv
from sklearn import linear_model

def getTrainData():
    return np.genfromtxt("spam_train.csv", delimiter=",")

def getTestData():
    return np.genfromtxt("spam_test.csv", delimiter=",")

def getLogiRegr(X, Y):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    print('Coef: ', logreg.coef_)
    print('Intercept: ', logreg.intercept_)
    return logreg

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(features, target, num_steps, learning_rate):
    intercept = np.ones((features.shape[0], 1))
    features = np.hstack((intercept, features))
    weights = np.zeros(features.shape[1])
    weights = np.matrix(weights).T
    
    for step in range(num_steps):
        scores = np.dot(weights.T, features.T)
        predictions = sigmoid(scores)
        # Update weights with gradient
#        output_error_signal = cross_entropy(target, predictions)
        gradient = diff_loss(target, predictions, features)
        weights -= learning_rate * gradient
#        learning_rate *= 0.95
        
    return weights

def cross_entropy(target, predictions):
    tmp = np.ones((target.shape[0], 1))
    
    sum_cross_entropy = -(np.dot(np.matrix(target), np.log10(np.matrix(predictions).T)))\
        + np.dot((tmp-target), np.log10(np.matrix(tmp-np.matrix(predictions).T)))
    
    return sum_cross_entropy

def diff_loss(target, predicts, features):
    sub = np.matrix(target) - predicts
    tmp = -np.dot(sub, features)
    
    return np.matrix(tmp).T

def outputToFile(result, is_mine):
    filename = "prediction.csv" if is_mine == 0 else "prediction2.csv"
    c = csv.writer(open(filename, "w", newline=''))
    output = []
    output.append("id")
    output.append("value")
    c.writerow(output)
    for i in range(0, len(result)):
        output = []
        output.append(str(i+1))
        output.append(result[i])
        c.writerow(output)
    
def main():    
    train_data = getTrainData()
    train_X = train_data[:, 1:train_data.shape[1]-1]
    train_Y = train_data[:, train_data.shape[1]-1]
    logiRegr = getLogiRegr(train_X, train_Y)
    
    test_data = getTestData()
    test_X = test_data[:, 1:test_data.shape[1]]
    predict_Y = []
    for i in range(0, test_X.shape[0]):
        tmp = test_X[i,:].dot(logiRegr.coef_.T) + logiRegr.intercept_
        predict_Y.append(sigmoid(tmp[0]))
#    print(predict_Y)
    
    result = []
    for i in range(0, len(predict_Y)):
        result.append(0 if predict_Y[i] < 0.5 else 1)
#    print(result)
    outputToFile(result, 0)
    
def my_main():
    learning_rate = 1e-5
    iteration_num = 3000
    train_data = getTrainData()
    train_X = train_data[:, 1:train_data.shape[1]-1]
    train_Y = train_data[:, train_data.shape[1]-1]
    w = logistic_regression(train_X, train_Y, iteration_num, learning_rate)
    print(w)
    
    test_data = getTestData()
    test_X = test_data[:, 1:test_data.shape[1]]
    predict_Y = []
    for i in range(0, test_X.shape[0]):
        tmp = np.dot(w[1:].T, test_X[i,:]) + w[0]
        predict_Y.append(sigmoid(tmp))
    print(predict_Y)
    
    result = []
    for i in range(0, len(predict_Y)):
        result.append(0 if predict_Y[i] < 0.5 else 1)
    print(result)
    outputToFile(result, 1)

#main()
my_main()
