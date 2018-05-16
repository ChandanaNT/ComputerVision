#Experimented with the CalTech 101 objects dataset
#Uses the bag of words representation of images for Naive Bayes classifier & Decision Tree

import os
import cv2
import numpy as np
from sklearn import tree
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

#Choose 3 categories
category_list = ['brain',  'ketch',  'chandelier']
dir = '/Users/chandanathippesh/Desktop/101_ObjectCategories/'


# SIFT does blurring anyway while detecting keypoints, so it's not necessary to blur the input images
sift = cv2.xfeatures2d.SIFT_create()

# Extract SIFT descriptors
Y = []
oc = 0
sift_keypoints = []
sift_descriptors = []
num_descriptors = []
for catg in category_list:
    folder_path = dir+catg
    for files in os.walk(folder_path):
        filenames_list = files[2]
        for image_name in filenames_list:
            complete_filename = folder_path+'/'+image_name
            img = cv2.imread(complete_filename)
            '''cv2.imshow(complete_filename,img)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()'''
            #Detect keypoints & descriptors for input image, keyp is a list, desc is a 2D numpy array
            keyp, desc = sift.detectAndCompute(img, None)
            sift_keypoints.append(keyp)
            sift_descriptors.append(desc)
            num_descriptors.append(len(desc))  #required to know which descriptor belongs to which image when we stack all descriptors together
            Y.append(oc)
    oc = oc + 1


#Stack up all the descriptors
l = len(sift_descriptors)
Z = np.vstack((sift_descriptors[0], sift_descriptors[1]))
for i in range(2,l):
    Z = np.vstack((Z,sift_descriptors[i]))       


# K Means Clustering
maximum_iterations = 15
epsilon = 1
num_clusters = 100 #Value of K
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maximum_iterations, epsilon)
#Vocabulary = Center
ret,label,center=cv2.kmeans(Z, num_clusters, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)


#Build Bag of Words Representation of Image
start = 0
X = []
for i in range(l):
    image_labels = label[start:start+num_descriptors[i]]
    start = start + num_descriptors[i]
    bogw = []
    for val in range(num_clusters):
        bogw.append((image_labels==val).sum())
    X.append(bogw)

X = np.array(X)
Y = np.array(Y)


#Split dataset into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# GAUSSIAN NAIVE BAYES 

#Training 
clf = GaussianNB()
clf.fit(X_train, Y_train.flatten())
Y_predicted = clf.predict(X_train)
print(accuracy_score(Y_train,Y_predicted)) #Train accuracy
#Inference
Y_predicted = clf.predict(X_test)
print(accuracy_score(Y_test,Y_predicted)) #Test accuracy
print(confusion_matrix(Y_test, Y_predicted))


#DESCISION TREE

#Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_train)
print(accuracy_score(Y_train,Y_predicted)) #Train accuracy
#Inference
Y_predicted = clf.predict(X_test)
print(accuracy_score(Y_test,Y_predicted)) #Test accuracy
print(confusion_matrix(Y_test, Y_predicted))


# Cross Validation
from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, X, Y, cv=6)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# Normalize the Data
X_normalised = normalize(X)
#Split dataset into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X_normalised,Y, test_size=0.2, random_state=42)

#GAUSSIAN NAIVE BAYES 

#Training 
clf = GaussianNB()
clf.fit(X_train, Y_train.flatten())
Y_predicted = clf.predict(X_train)
print(accuracy_score(Y_train,Y_predicted)) #Train accuracy
#Inference
Y_predicted = clf.predict(X_test)
print(accuracy_score(Y_test,Y_predicted)) #Test accuracy
print(confusion_matrix(Y_test, Y_predicted))



# DECISION TREE 

#Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_train)
print(accuracy_score(Y_train,Y_predicted)) #Train accuracy
#Inference
Y_predicted = clf.predict(X_test)
print(accuracy_score(Y_test,Y_predicted)) #Test accuracy
print(confusion_matrix(Y_test, Y_predicted)) 

#Try using class_weight = "balanced" to assign weights to classes depending on # of samples in the classes
clf = tree.DecisionTreeClassifier(class_weight="balanced")
clf = clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_train)
print(accuracy_score(Y_train,Y_predicted)) #Train accuracy
#Inference
Y_predicted = clf.predict(X_test)
print(accuracy_score(Y_test,Y_predicted)) #Test accuracy

