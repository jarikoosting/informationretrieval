#!/usr/bin/python3

import nltk.classify
import string # for removing punctuation
from nltk.tokenize import word_tokenize
from nltk.probability import LaplaceProbDist
from nltk.corpus import stopwords
from featx import bag_of_words, high_information_words
from classification import precision_recall

from random import shuffle
from os import listdir # to read files
from os.path import isfile, join # to read files
import sys

# return all the filenames in a folder
def getFilenamesInFolder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

def removeFileName(trainData):

    modifiedData = list()
    
    for el in trainData:
        modifiedData.append(el[:2])

    return modifiedData

def splitDataSet(data, folds=2):

    shuffle(data) # randomise dataset before splitting into train and test

    splittedDataSet = []
    countReviews = len(data)
    trainAmount = int(countReviews * 0.9)

    # Append train data
    splittedDataSet.append(data[:trainAmount])

    # Append test data
    splittedDataSet.append(data[trainAmount:])
    print("\n Split datasets to trainset and testset..")

    return splittedDataSet
    
    return splittedData

def train(trainData):

    classifier = nltk.classify.NaiveBayesClassifier.train(trainData, estimator=LaplaceProbDist)
    return classifier

def removePunctiation(tokens):

    no_punctuation = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    return [s for s in no_punctuation if s]

def removeStopWords(tokens):

    # Get English stopwords
    stop_words = set(stopwords.words("english"))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '<', '>', '</br>', 'br', '!',"''", '/', '``'])

    return [w for w in tokens if w not in stop_words]

def getTrainData(categories):

    trainData = list()

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
            #trainData.append((bag, category, f))

            # Break after 50 files, so we can test better 
            num_files+=1
            if num_files>=400: 
               break

    print("  Total, %i files read" % (len(trainData)))

    return trainData

def high_information(feats, categories):

    " Code by Antonio "

    print("\n##### Obtaining high information words...")

    labelled_words = [(category, []) for category in categories]

    # 1. convert the formatting of our features to that required by high_information_words
    from collections import defaultdict
    words = defaultdict(list)
    all_words = list()
    for category in categories:
        words[category] = list()

    for feat in feats:
        category = feat[1]
        bag = feat[0]
        for w in bag.keys():
            words[category].append(w)
            all_words.append(w)

    labelled_words = [(category, words[category]) for category in categories]

    # 2. calculate high information words
    high_info_words = set(high_information_words(labelled_words))

    print("  Number of words in the data: %i" % len(all_words))
    print("  Number of distinct words in the data: %i" % len(set(all_words)))
    print("  Number of distinct 'high-information' words in the data: %i" % len(high_info_words))

    return high_info_words

def calculate_f(precisions, recalls):
	f_measures = {}

	for category in precisions:
		f_measures[category] = 2 * precisions[category] * recalls[category] / (precisions[category] + recalls[category])

	return f_measures

def evaluation(classifier, testData, categories):
	print ("\n##### Evaluation...")
	print("  Accuracy: %f" % nltk.classify.accuracy(classifier, testData))
	precisions, recalls = precision_recall(classifier, testData)
	f_measures = calculate_f(precisions, recalls)  

	print(" |-----------|-----------|-----------|-----------|")
	print(" |%-11s|%-11s|%-11s|%-11s|" % ("category","precision","recall","F-measure"))
	print(" |-----------|-----------|-----------|-----------|")
	for category in categories:
		if precisions[category] is None:
			print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
		elif (precisions[category] == 0) and (recalls[category] == 0):
			print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], "NA"))
		else:
			print(" |%-11s|%-11f|%-11f|%-11f|" % (category, precisions[category], recalls[category], f_measures[category]))
	print(" |-----------|-----------|-----------|-----------|")
	
    
    
def main():

    categories = ["neg","pos"]

    trainData = getTrainData(categories)

    high_info_words = high_information(trainData, categories)

    splittedDataSet = splitDataSet(trainData)

    trainData = splittedDataSet[0]
    testData = splittedDataSet[1]

    trainData = removeFileName(trainData)
    classifier = train(trainData)

    evaluation(classifier, testData, categories)

    #for sentence, label, filename in testData:
        
    #    print(filename, classifier.classify(sentence))


if __name__ == "__main__":

    main()
