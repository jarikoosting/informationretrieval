#!/usr/bin/python3

import nltk.classify
import string # for removing punctuation
from nltk.tokenize import word_tokenize
from nltk.probability import LaplaceProbDist
from nltk.corpus import stopwords
from featx import bag_of_words, high_information_words

from random import shuffle
from os import listdir # to read files
from os.path import isfile, join # to read files
import sys

# return all the filenames in a folder
def getFilenamesInFolder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

def splitDataSet(data, folds=10):

    shuffle(data) # randomise dataset before splitting into train and test

    # Calculate how big 1/n part of the list has to be
    chunk = round(len(data) / folds)
    
    # divide feats into n cross fold sections
    splittedData = []
    for n in range(folds):

        length = n * chunk

        # Get the test feats
        testData = data[length:length+chunk+1]

        # Get all the feats without the test feat, these are the train feats
        trainData = data[:length] + data[(length + chunk + 1):]

        splittedData.append((trainData, testData))
    
    print("\n Splitting datasets..")
    return splittedData

def train(trainData):
    print(trainData)
    classifier = nltk.classify.NaiveBayesClassifier.train(trainData, estimator=LaplaceProbDist)
    return classifier

def removePunctiation(tokens):

    no_punctuation = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    return [s for s in no_punctuation if s]

def removeStopWords(tokens):

    # Get English stopwords
    stop_words = set(stopwords.words("english"))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '<', '>', '</br>', 'br', '!',"''"])

    return [w for w in tokens if w not in stop_words]

def getTrainData():

    trainData = list ()
    categories = ["neg","pos"]

    for category in categories:

        files = getFilenamesInFolder('reviews/' + category)
        num_files=0

        for f in files:
            data = open('reviews/' + category + '/' + f, 'r', encoding='UTF-8').read()
            tokens = word_tokenize(data)

            no_punctuation = removePunctiation(tokens)

            # Don't use stop words 
            filtered_sentence = removeStopWords(tokens)

            bag = bag_of_words(filtered_sentence)
            trainData.append((bag, category))

            # Break after 50 files, so we can test better 
            num_files+=1
            if num_files>=50: 
               break

    print("  Total, %i files read" % (len(trainData)))

    return trainData

def main():

    trainData = getTrainData()
    splittedDataSet = splitDataSet(trainData)

    for trainData, testData in splittedDataSet:
        classifier = train(trainData)

        for sentence, label in testData:
            print (classifier.classify(sentence))

if __name__ == "__main__":

    main()
