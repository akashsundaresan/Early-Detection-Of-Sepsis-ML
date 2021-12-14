from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'predict/index.html',{'title':'cybertehz'})

def login(request):
    return render(request,'predict/login.html',{'title':'login'})

def signup(request):
    return render(request,'predict/signup.html',{'title':'signup'})

def uploadfiles(request):
    return render(request,'predict/upload.html',{'title':'uploadfiles'})

def report1(request):
    return render(request,'predict/report1.html',{'title':'report'})

def result1(request):
    return render(request,'predict/result1.html',{'title':'result'})

def result2(request):
    return render(request,'predict/result2.html',{'title':'result'})


def report2(request):
    return render(request,'predict/report2.html',{'title':'report'})


def analyze(request):
    return render(request,'predict/analyze.html',{'title':'report'})




def loadCsv(filename):
    lines=csv.reader(open('data.csv'))
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
 
    filename='data.csv'
    splitRatio =0.80
  
    trainingSet,testSet = splitdataset(dataset,splitRatio)
    print("spilit {0} rows into train = {1} and test = {2} rows".format(len(dataset),len(trainingSet),len(testSet)))
    summaries = summarizeByClass(trainingSet)
    predictions= getPredicitions(summaries,testSet)
    cm=confusion_matrix(float(testSet),predictions) 
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy: {0}%'.format(accuracy))

