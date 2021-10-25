import numpy as np
from math import sqrt
from math import pi
from math import exp

class NaiveBayes():
    def __init__(self, train, test):
        summary = summarizeByClass(train)
        predictions = list()
        for row in test:
            predicitons.append(predict(summary, row))
        return predictions
    
    def predict(self, summary, row):
        probabiities = calcClassProbability(summary, row)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel
    
    def classSeparate(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            classValue = vector[-1]
            if (classValue not in separated):
                separated[classValue] = list()
            separated[classValue].append(vector)
        return separated
    
    def summarizeDataset(self, dataset):
        summaries = [(mean(col), stdDev(col), len(col)) for col in zip(*dataset)]
        del(summaries[-1])
        return summaries
    
    def summarizeByClass(self, dataset):
        separated = self.classSeparate(dataset)
        summaries = dict()
        for classValue, row in separated.items():
            summaries[classValue] = self.summarizeDataset(row)
        return summaries
    
    def calcClassProbability(self, summaries, row):
        totalRows = sum([summaries[label][0][-1] for label in summaries])
        probabilities = dict()
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = summaries[classValue][0][-1] / float(totalRows)
            for i in range(len(classSummaries)):
                mean, stddev, count = classSummaries[i]
                probabilities[classValue] *= calcGaussianProbability(row[i], mean, stddev)
        return probabilities
    
def mean(data):
    return sum(data)/float(len(data))

def stdDev(data):
    avg = mean(data)
    variance = sum([(x-avg)**2 for x in data]) / float(len(data)-1)
    return sqrt(variance)

def calcGaussianProbability(x, mean, stddev):
    return (1 / (sqrt(2*pi) * stddev)) * exp(-((x-mean)**2 / (2 * stddev**2)))
