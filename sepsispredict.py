import csv
import math
import random
from sklearn.metrics import confusion_matrix


def loadCsv(filename):
    lines=csv.reader(open(filename))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
        print('Data Training',i)
    return dataset

def splitdataset(dataset,splitRatio):
    trainSize=int(len(dataset)*splitRatio)
    trainSet=[]
    copy=list(dataset)
    while len(trainSet) < trainSize:
        index=random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]

def separateByClass(dataset):
    separated={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg=mean(numbers)
    variance=sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries=[(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries ={}
    for classValue, instances in separated.items():
        summaries[classValue]=summarize(instances)
    return summaries

def calculateProbability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilities(summaries,inputVector):
    probabilities={}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue]=1
        for i in range(len(classSummaries)):
            mean,stdev=classSummaries[i]
            x=inputVector[i]
            probabilities[classValue]*=calculateProbability(x,mean,stdev)
        return probabilities

def predict(summaries, inputVector):
    probabilities=calculateClassProbabilities(summaries,inputVector)
    bestLabel,bestProb =None,-1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability >bestProb:
            bestProb=probability
            bestLabel=classValue
    return bestLabel

def getPredicitions(summaries,testSet):
    predictions=[]
    for i in range(len(testSet)):
        result=predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet,predictions):
        correct =0
        for x in range(len(testSet)):
            if testSet[x][-1]==predictions[x]:
                correct+=1
        return (correct/float(len(testSet)))*100.0
def main():
 
    filename='finaldata.csv'
    splitRatio =0.80
   # dataset = [[3.393533211,2.331273381,0],
	#[3.110073483,1.781539638,0],
	#[1.343808831,3.368360954,0],
	#[3.582294042,4.67917911,0],
#	[2.280362439,2.866990263,0],
#	[7.423436942,4.696522875,1],
#	[5.745051997,3.533989803,1],
#	[9.172168622,2.511101045,1],
#	[7.792783481,3.424088941,1],
#	[7.939820817,0.791637231,1]]
    dataset=loadCsv(filename)
    trainingSet,testSet = splitdataset(dataset,splitRatio)
    print("spilit {0} rows into train = {1} and test = {2} rows".format(len(dataset),len(trainingSet),len(testSet)))
    summaries = summarizeByClass(trainingSet)
    predictions= getPredicitions(summaries,testSet)
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy: {0}%'.format(accuracy))

main()