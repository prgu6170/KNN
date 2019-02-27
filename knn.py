#!/usr/bin/env python
# coding: utf-8
# Homework 1 - KNN
## CSCI 5622 - Spring 2019

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle
from collections import Counter
import sklearn.neighbors


class KNNClassifier:

    def __init__(self, X, y, k=5):
        """
        Initializes custom KNN classifier
        PARAMETERS
        X -  training data features
        y -  training data answers
        k - the number of nearest neighbors to consider for classification
        """
        self._model = sklearn.neighbors.BallTree(X)
        self._y = y
        self._k = k
        self._x = X
        self._counts = self.getCounts()

    def getCounts(self):
        """
        Creates a dictionary storing the counts of each answer class found in y
        returns counts - a dictionary of counts of answer classes
        """

        unique, count = np.unique(self._y, return_counts=True)
        counts = dict(zip(unique, count))
        return counts

    def majority(self, indices):
        """
        Given indices, reports the majority label of those points.
        For a tie, report the most common label in the data sets
        returns label - the majority label of our neighbors
        """
        count_neighbor = []
        for item in indices:
            count_neighbor.append(self._y[item])
        counter = Counter(count_neighbor)
        most_common = [counter.most_common()[0][0]]
        higher = counter.most_common()[0][1]
        i = 0
        for item in counter.most_common():
            if counter.most_common()[i][1] == higher:
                most_common.append(counter.most_common()[i][0])
                i += 1
            else:
                break
        total_counts = self.getCounts()
        label = most_common[0]
        highest = total_counts[most_common[0]]
        j = 0
        for item in most_common:
            if total_counts[most_common[j]] > highest:
                label = most_common[j]
        return label

    def classify(self, given_point):
        """
        Given a new data point, classifies it according to the training data X and our number of neighbors k into the appropriate class in the training answers y
        returns ans - the predicted classification
        """
        dimensions = len(given_point)
        dist, ind = self._model.query(np.reshape(given_point, (-1, dimensions)), k=self._k)
        ans = self.majority(ind[0])
        return ans

    def confusionMatrix(self, testX, testY):
        """
        Generates a confusion matrix for the given test set
        returns C - an N*N np.array of counts, where N is the number of classes in the classifier
        """
        C = np.zeros((len(self._counts), len(self._counts)), dtype=int)
        i = 0
        class_number = {}
        for key, value in self._counts.items():
            class_number[key] = i
            i += 1
        i = 0
        prediction_array = []
        wrong_predictions = []
        for x in testX:
            prediction = self.classify(x)
            prediction_array.append(prediction)
            C[class_number[testY[i]]][class_number[prediction]] += 1
            if class_number[testY[i]] != class_number[prediction]:
                wrong_predictions.append(class_number[testY[i]])
            i += 1
        return C, wrong_predictions

    def accuracy(self, C):
        """
        Generates an accuracy score for the classifier based on the confusion matrix
        returns score - an accuracy score
        """
        score = np.sum(C.diagonal()) / C.sum()
        return score


class Numbers:
    def __init__(self):
        # load data from sklearn
        self.digits = sklearn.datasets.load_digits()
        self.split = 80 * len(self.digits.data) / 100
        self.data = self.digits.data
        self.target = self.digits.target
        self.random_data, self.random_target = shuffle(self.data, self.target)
        self.train_x = self.random_data[:int(self.split), :]  # A 2D np.array of training examples, REPLACE
        self.train_y = self.random_target[:int(self.split)]  # A 1D np.array of training answers, REPLACE
        self.test_x = self.random_data[int(self.split):, :]  # A 2D np.array of testing examples, REPLACE
        self.test_y = self.random_target[int(self.split):]  # A 1D np.array of testing answers, REPLACE

    def report(self):
        """
        Reports information about the dataset using the print() function
        """
        print(f"Size of train set : {len(self.train_x)}")
        print(f"Size of each datapoint in X: {len(self.train_x[0])}")
        print(f"Size of test set : {len(self.test_x)}")

    def classify(self, k_value):
        """
        Creates a classifier using the training data and generate a confusion matrix for the test data
        """
        model = KNNClassifier(self.train_x, self.train_y, k_value)
        conf_mat, incorrect_predictions = model.confusionMatrix(self.test_x, self.test_y)
        unique, count = np.unique(self.test_y, return_counts=True)
        counts = dict(zip(unique, count))
        return conf_mat, model.accuracy(conf_mat), incorrect_predictions

    def viewDigit(self, digitImage):
        """
        Displays an image of a digit
        """
        plt.gray()
        plt.matshow(self.digits.images[digitImage])
        plt.show()

        
num = Numbers()
num.report()
matrix, accuracy, incorrect = num.classify(5)
print("\nConfusion matrix for K=5")
print(matrix)
print("\nAccuracy with split 80/20 and K=5")
print(accuracy)


for item in incorrect:
    num.viewDigit(item)

class Numbers2:
    def __init__(self, trainPercentage):
        # load data from sklearn
        self.digits = sklearn.datasets.load_digits()
        self.split = trainPercentage * len(self.digits.data) / 100
        self.data = self.digits.data
        self.target = self.digits.target
        self.random_data, self.random_target = shuffle(self.data, self.target)
        self.train_x = self.random_data[:int(self.split), :]  # A 2D np.array of training examples,
        self.train_y = self.random_target[:int(self.split)]  # A 1D np.array of training answers,
        self.test_x = self.random_data[int(self.split):, :]  # A 2D np.array of testing examples,
        self.test_y = self.random_target[int(self.split):]  # A 1D np.array of testing answers,

    def report(self):
        """
        Reports information about the dataset
        """
        print(f"Size of train set : {len(self.train_x)}")
        print(f"Size of each datapoint in X: {len(self.train_x[0])}")
        print(f"Size of test set : {len(self.test_x)}")

    def classify(self, k_value):
        """
        Creates a classifier using the training data and generate a confusion matrix for the test data
        """
        model = KNNClassifier(self.train_x, self.train_y, k_value)
        conf_mat, incorrect_predictions = model.confusionMatrix(self.test_x, self.test_y)
        unique, count = np.unique(self.test_y, return_counts=True)
        counts = dict(zip(unique, count))
        return conf_mat, model.accuracy(conf_mat), incorrect_predictions

    def viewDigit(self, digitImage):
        """
        Displays an image of a digit
        """
        plt.gray()
        plt.matshow(self.digits.images[digitImage])
        plt.show()
        
        
k = list(range(1, 20))
accuracy_list = []
for each_k in k:
    cnf_mtrx, acc, wrng_prd = num.classify(each_k)
    accuracy_list.append(acc)
plt.plot(k, accuracy_list)
plt.xlabel("Value of K")
plt.ylabel("Accuracy of prediction")
plt.title("K vs.,Accuracy graph")
plt.show()

k = 5
accuracy_list = []
percent = np.arange(50, 85, 5)
for prcnt in percent:
    num = Numbers2(prcnt)
    print(f"Percent split: {prcnt}")
    num.report()
    cnf_mtrx, acc, wr_pred = num.classify(k)
    accuracy_list.append(acc)
plt.plot(percent, accuracy_list)
plt.xlabel("Percentage split")
plt.ylabel("Accuracy of prediction")
plt.title("Percentage split vs.,Accuracy graph")
plt.show()
