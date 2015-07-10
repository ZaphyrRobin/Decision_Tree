
# coding: utf-8

# In[1]:

#import Spark and MLlib packages
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils

#import data analysis packages
import numpy as np
import pandas as pd
import sklearn

from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from numpy import array

from sklearn import tree
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

#for sklearn decision tree pdf plotting
from sklearn.externals.six import StringIO
import pydot

#import data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

#misc packages
from __future__ import division
from __future__ import print_function


# In[2]:

#I.Load dataset
mem = Memory("./mycache")

#using decoration to pass file to memory
@mem.cache
def get_data():
    data = load_svmlight_file("/usr/local/spark/data/mllib/sample_libsvm_data.txt")
    return data[0], data[1]

x, y = get_data()


# In[3]:

#Have to convert to dense array to fit the model
dense_x = x.toarray()

#Split the training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(dense_x, y, test_size=0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[4]:

#Training the model
ID3 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5).fit(X_train, Y_train)
CART = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=5).fit(X_train, Y_train)


# In[5]:

def cal_model_accuracy(list):
    for i, clf in enumerate(list):
        predicted = clf.predict(X_test)
        expected = Y_test
    
        #compare results
        accuracy = metrics.accuracy_score(expected, predicted)
        if i==0: print("ID3 accuracy is {}".format(accuracy))
        else:    print("CART accuracy is {}".format(accuracy))

l_list = (ID3, CART)
cal_model_accuracy(l_list)


# In[7]:

#Generate pdf file of decision tree
dot_data = StringIO()
tree.export_graphviz(CART, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("CART_decision_tree.pdf")


# In[8]:

#IV Use MLlib
sc = SparkContext("local", "Decision_tree")


# In[9]:

data = MLUtils.loadLibSVMFile(sc, '/usr/local/spark/data/mllib/sample_libsvm_data.txt')


# In[11]:

data.take(1)


# In[12]:

#Split the training set and test set
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# In[13]:

#Training model
ID3_model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='entropy', maxDepth=5, maxBins=32)
CART_model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)


# In[14]:

#Predication
def cal_mllib_accuracy(list):
    for i, clf in enumerate(list):
        #prediction with the features
        predictions = clf.predict(testData.map(lambda x: x.features))
        #append with lables first then features
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        
        accuracy = labelsAndPredictions.filter(lambda (v, p): v == p).count()/testData.count()
    
        #compare results
        
        if i==0: print("PySpark ID3 accuracy is {}".format(accuracy))
        else:    print("PySpark CART accuracy is {}".format(accuracy))
            
cal_mllib_accuracy((ID3_model, CART_model))

