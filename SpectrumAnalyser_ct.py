import sys
import string
import random
import time

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Spectrum():
    #####初始化类实例，创建一个空的testCases列表。
    # printCode = True
    def __init__(self):
        self.testCases = []

    ####################从给定的文件中读取测试用例，并将其添加到testCases列表中。
    def produceTestCases(self, noOfTestCasesp, groundTruthFilename):
        self.noOfTestCases = noOfTestCasesp
        with open("00programs/"+groundTruthFilename) as f:
            groundTruthCode = f.read()
        printCode = True

        if printCode: print("...produce TestCases start******")
        for ii in range(self.noOfTestCases):
            gtGlobals = {}
            exec(groundTruthCode,gtGlobals)
            del gtGlobals["__builtins__"]
            self.testCases.append([gtGlobals["inputTC"],gtGlobals["outputTC"]])
            # if printCode: print(gtGlobals["inputTC"]," ",gtGlobals["outputTC"])
        if printCode: print("...produce TestCases end******")
    #############从给定的文件中读取程序代码，并执行程序。通过执行程序，获得程序的特征字典和执行结果。
    def executeFromFile(self, progUnderTestFilename):
        with open("00programs/"+progUnderTestFilename) as f:
            self.programText = f.readlines()

        printCode = True

        programToRun = "".join(self.programText)
        progGlobals = {}
        progGlobals["inputValue"] = self.testCases[0][0] #arbitrary test case

        # if printCode: print(progGlobals["inputValue"])

        exec(programToRun, progGlobals)
        self.featureDict = {}
        idx = 0

        if printCode: print("...execute FromFile start******")

        for var in progGlobals.keys():
            if isinstance(progGlobals[var],int):
                self.featureDict.update({var: idx})
                idx += 1

        self.noOfLines = len(self.programText)
        self.noOfFeatures = len(self.featureDict)
        self.execute()

        if printCode: print("...execute FromFile end******")
    ##################执行程序的主要逻辑，计算每个测试用例在每行代码执行后的特征向量，并将其存储在spectrum数组中。      
    def execute(self):
        self.spectrum = np.zeros((self.noOfTestCases,self.noOfLines,self.noOfFeatures),np.int16)
        printCode = True

        # if printCode: print(self.noOfTestCases,self.noOfLines,self.noOfFeatures,self.programText)
        ## test case, last line executed, collection of variables by line
        for tcIdx, sample in enumerate(self.testCases):
            # if printCode: print("###############################################################################")
            # if printCode: print("##### test case {:d}".format(tcIdx))
            for maxLine in range(0,self.noOfLines):
                # if printCode: print("%%%%% line up to {:d}".format(maxLine+1))
                programToRun = ""
                for line in range(0,maxLine+1):
                    programToRun += "{:s}".format(self.programText[line]) # wibble could replace with join

                progGlobals = {}
                progGlobals["inputValue"] = sample[0]
                exec(programToRun, progGlobals)
                if "=" in  self.programText[maxLine]:
                    varName = self.programText[maxLine].split("=",1)[0].strip()
                    if varName in self.featureDict:
                        val = progGlobals[varName]
                if maxLine>0:
                    self.spectrum[tcIdx,maxLine,:] = self.spectrum[tcIdx,maxLine-1,:] #copy down previous line
                    if varName in self.featureDict:
                        self.spectrum[tcIdx,maxLine,self.featureDict[varName]] = val
        # if printCode: print(programToRun, self.spectrum)
        # # remove empty features
        # indices = []
        # for feature in range(self.noOfFeatures):
        #     featureArray = self.spectrum[:,:,feature].flatten()
        #     if (np.all(featureArray==0)):
        #         indices.append(feature)
        # #self.spectrum = np.delete(self.spectrum, indices, axis=2)
        # self.numberOfFeatures = np.shape(self.spectrum)[2] #wibble problem here

    ##############对spectrum数组进行分析，使用朴素贝叶斯分类器（CategoricalNB、GaussianNB或MultinomialNB）计算每行代码的得分
    def analyseSpectrum(self):
        scores = [0.0]
        printCode = True

        if printCode: print("...analyse Spectrum start******")
        for line in range(self.noOfLines):
            #print("############## line number: {:d}".format(line))
            Xtrain = []
            Ytrain = []
            Xtest = []
            Ytest = []
            for idx,tc in enumerate(self.testCases):
                #print("**** test case {:d}".format(idx))
                instance = self.spectrum[idx,line,:]
                if random.random()>0.1:
                    Xtrain.append(instance)
                    Ytrain.append(tc[1])
                else:
                    Xtest.append(instance)
                    Ytest.append(tc[1])
            # clf = GaussianNB()
            # clf = DecisionTreeClassifier() # Decision Trees
            # clf = RandomForestClassifier() # Ensemble Learning
            clf = LogisticRegression() # Ensemble Learning
            # clf = SVC(kernel='linear') # Support Vector Machine, SVM
            # clf = LinearRegression() # Linear Regression
            # clf = KNeighborsClassifier() # K-Nearest Neighbors，KNN
            clf.fit(Xtrain,Ytrain)
            scores.append(clf.score(Xtest,Ytest))

        if printCode: print("...Spectrum algorithm end******")
        toIgnore = [] #comments, whitespace, etc.
        for idx,statement in enumerate(self.programText):
            if statement.strip()=="" or statement.strip()[0]=="#" or statement.strip()[0:6]=="import":
                toIgnore.append(idx+1)
        for i in toIgnore:
            if i>0:
                scores[i] = scores[i-1] #don't update - no new information by definition
        # print(["{0:0.4f}".format(pp) for pp in scores])
        return scores

    ################根据得分结果找到可能的错误行，并返回预测的错误行号。
    def findError(self,decisionCriterion):
        # decisionCriterion is "largest" or "first"
        printCode = True
        
        profile = self.analyseSpectrum()
        if printCode: print("...finding error start******")
        if printCode: print(["{0:0.4f}".format(pp) for pp in profile])

        predictionMade = False
        prediction = None
        diffs = [0.0]
        if decisionCriterion=="largest":
            largestDiff = -1.0
            argLargestDiff = -999
            for i in range(1,len(profile)):
                diff = profile[i-1]-profile[i]
                diffs.append(diff)
                if diff > largestDiff:
                    largestDiff = diff
                    argLargestDiff = i
            predictionMade = argLargestDiff!=-999
            prediction = argLargestDiff-1
        elif decisionCriterion=="first":
            for i in range(1,len(profile)):
                diff = profile[i]-profile[i-1]
                diffs.append(diff)
                if diff<-0.0025:
                    if not predictionMade:
                        prediction = i-1
                    predictionMade = True
        else:
            print("Error: decision criterion not recognised")
            sys.exit()

        if (printCode):
            for i in range(len(self.programText)):
                if i==prediction and predictionMade:
                    print("** {:0.4f} \t\t {:0.4f} \t {:s}".format(profile[i+1],diffs[i+1],self.programText[i].strip()))
                else:
                    print("{:0.4f} \t\t {:0.4f} \t {:s}".format(profile[i+1],diffs[i+1],self.programText[i].strip()))
        if printCode: print("...finding error end******")
        return predictionMade, prediction #the prediction of the line where the error is

    ################### 多次运行findError方法，并统计预测结果。
    def runFindErrorMultipleTimes(self,numberOfRepeats,decisionCriterion):
        predictions = np.zeros((self.noOfLines),dtype=int)
        for rep in range(numberOfRepeats):
            predictionMade, errorPrediction = self.findError(decisionCriterion)

            if (predictionMade):
                predictions[errorPrediction] += 1
        if all(pre==0 for pre in predictions): #have *any* predictions been made
            predictionMade = False
        else:
            predictionMade = True
        bestPrediction = np.argmax(predictions)
        # print()
        answer = ""
        for i in range(self.noOfLines):
            if i==bestPrediction and predictionMade:
                print("** {:d} \t {:s}".format(predictions[i], self.programText[i].strip()))
                answer += "** {:d} \t {:s}".format(predictions[i], self.programText[i].strip())
            else:
                print("{:d} \t {:s}".format(predictions[i], self.programText[i].strip()))
                answer += "{:d} \t {:s}".format(predictions[i], self.programText[i].strip())
        return answer, bestPrediction, np.max(predictions), predictions

    ###################运行整个程序，包括生成测试用例、执行程序和查找错误。
    def runMainProgram(self,filenamePUT,filenameGT,numberOfTestCases,numberOfRuns,decisionCriterion):
        time_start = time.time()
        self.produceTestCases(numberOfTestCases,filenameGT)
        self.executeFromFile(filenamePUT)
        finalProfile, bestPrediction, noOfCorrectPredictions, predictions = self.runFindErrorMultipleTimes(numberOfRuns,decisionCriterion)
        time_end = time.time()
        print('time cost:', time_end - time_start)
        return bestPrediction, noOfCorrectPredictions, predictions

    ###################判断程序在多次运行中是否呈现近单调性，即特征向量的变化趋势是否单调递增
    def isMonotonic(self,filenamePUT,filenameGT,numberOfTestCases,numberOfRuns):
        self.produceTestCases(numberOfTestCases,filenameGT)
        self.executeFromFile(filenamePUT)
        numberNearMonotonic = 0
        for rep in range(numberOfRuns):
            profile =  self.analyseSpectrum()
            #print(["{0:0.4f}".format(pp) for pp in profile])
            print(["{0:0.4f}".format(pp) for pp in np.diff(profile)])
            numberNearMonotonic += 1 if all(dd>-0.01 for dd in np.diff(profile)) else 0
            if all(dd>-0.01 for dd in np.diff(profile)) : print("NM")
            print()
        print("NNM: ",numberNearMonotonic)
        return numberNearMonotonic

    #################可视化谱覆盖结果，根据测试用例的输出和特征向量绘制柱状图。
    def visualiseSpectrum(self):
        outputs = set()
        for tc in self.testCases:
            outputs.add(tc[1])
        minTestCaseOutput = min(outputs)
        maxTestCaseOutput = max(outputs)
        noOfTestCaseOutputs = maxTestCaseOutput-minTestCaseOutput+1
        charts = np.empty( (maxTestCaseOutput+1,self.spectrum.shape[2]), dtype=Chart )

        for tcOutput in range(maxTestCaseOutput+1):
            for ffIdx in range(self.spectrum.shape[2]):
                firstFeature = min(self.spectrum[:,-1,ffIdx])
                lastFeature = max(self.spectrum[:,-1,ffIdx])
                ch = Chart(tcOutput,firstFeature,lastFeature)
                charts[tcOutput,ffIdx] = ch

        for tcIdx,tc in enumerate(self.testCases):
            for ffIdx in range(self.spectrum.shape[2]):
                value = self.spectrum[tcIdx,-1,ffIdx]
                tcOutput = tc[1]
                charts[tcOutput,ffIdx].add(value)

        fig,axes = plt.subplots(maxTestCaseOutput+1-minTestCaseOutput,self.spectrum.shape[2])
        for ri,row in enumerate(axes):
            for ci,col in enumerate(row):
                charts[ri+minTestCaseOutput,ci].display(col)

        # for tcIdx in range(minTestCaseOutput,maxTestCaseOutput+1):
        #     for ffIdx in range(self.spectrum.shape[2]):
        #         subplotIdx = (tcIdx-minTestCaseOutput)*self.spectrum.shape[2] + ffIdx + 1
        #         fig.add_subplot(maxTestCaseOutput+1-minTestCaseOutput,self.spectrum.shape[2],subplotIdx)
        #         charts[tcIdx,ffIdx].display()
        plt.show()
                
        # self.spectrum is numpy array indexed by [testcase, line, feature]
        # self.testCases is list of [input, output] pairs

    ####################可视化程序在多次运行中的谱覆盖结果，并绘制变化曲线图。
    def visualiseMonotonicity(self,PUTfilename,GTfilename,noOfRuns):
        self.produceTestCases(noOfRuns,GTfilename)
        self.executeFromFile(PUTfilename)
        profile = self.analyseSpectrum()
        print(profile)
        # Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
        # Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]

        pt = self.programText.copy()
        pt.insert(0,"[before execution]")
        pt = [str(a)+". "+b for a,b in zip(range(0,len(pt)),pt)]
        plt.plot(pt, profile, color='black', marker='o')
        #plt.title('Progress measure by line')
        plt.xlabel('Program Line')
        plt.ylim([-0.05, 1.05])
        plt.xticks(rotation=90,ha='left')
        plt.tight_layout() 
        plt.ylabel('Progress Measure')
        plt.show()

####################类用于存储每个测试用例和特征向量之间的关系
class Chart:
    def __init__(self,testCaseLabelp,firstFeature,lastFeature):
        self.testCaseLabel = testCaseLabelp
        #numberOfBars = lastFeature-firstFeature
        self.labels = range(firstFeature,lastFeature+1)
        self.values = [0]*len(self.labels)
        
    ##################################################################################################
    def add(self,value):
        self.values[list(self.labels).index(value)] +=1
        
    ##################################################################################################
    def display(self,ax):
        #plt.style.use('ggplot')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.bar(self.labels, self.values, color='green')
        #label_pos = [i for i, _ in enumerate(label)]
        #ax.xticks(label_pos, label)

        
##########################runAllExperiments和runAllMonotonicTests函数分别用于运行实验和单调性测试，并返回结果的字符串表示。
def runAllExperiments(experimentsFile,noOfTCs,noOfRepetitions):
    # experiments file: csv with program name, programUT, programGT,differenceString,decisionType,errorLine)
    with open(experimentsFile) as f:
        experimentsList = f.readlines()
    ans = "\\begin{tabular}{llllll}\n"
    ans += "Program & Change Made & Method & Error Found & Number of Correct Predictions/"+str(noOfRepetitions)+" & Rank of Best Prediction \\\\ \n" 
    ans += "\\hline\n"
    for exper in experimentsList:
        print("********")
        experiment = exper.split(",")
        s = Spectrum()
        result = s.runMainProgram(experiment[1].strip(),experiment[2].strip(),int(noOfTCs),int(noOfRepetitions),experiment[4].strip())
        rankOfBestPrediction = (len(result[2])-scipy.stats.rankdata(result[2],method="max"))[int(experiment[5])]
        ans += experiment[0] + " & "+ experiment[3] + "&" + experiment[4].strip() + "&" + ("yes" if int(result[0])==int(experiment[5]) else "no") \
            + " & " + ((str(result[1])+"/"+str(noOfRepetitions)) if int(result[0])==int(experiment[5]) else "\\ ") \
            + " & " + str(rankOfBestPrediction+1) + "/" + str(len(result[2])) + "\\\\ \n" #+1 because ranks start at zero
    ans += "\\end{tabular}\n"
    return ans

###########################################################################################################
def runAllMonotonicTests(programsList,noOfTCs,noOfRepetitions): #programs list is list of pairs (PUT,GT)
    ans = "\\begin{tabular}{ll}\n"
    ans += "Program Name &  Number of Near-monotonic Vectors \\\\ \n" 
    ans += "\\hline\n"
    for [filenamePUT,filenameGT] in programsList:
        s = Spectrum()
        results = s.isMonotonic(filenamePUT,filenameGT,noOfTCs,noOfRepetitions)
        ans += filenamePUT + " & " + str(results) + "/" + str(noOfRepetitions) + " \\\\ \n" 
    ans += "\\end{tabular}\n"
    return ans
    #CWP


###################### older #####
# s = Spectrum()
# s.produceTestCases(10000, "VowelCounter_GT.py")
# s.produceTestCases(10000, "Get_Middle_Number_GT.py")
# s.produceTestCases(10000, "VowelCounter_GT.py")
# print(s.testCases)
# s.executeFromFile("VowelCounter.py")
# #print(s.spectrum[0,:,:]) #example
# #profile = s.analyseSpectrum()
# #print(["{0:0.4f}".format(pp) for pp in profile])
# print("############################################################################################")
#s.findError()
# s.runFindErrorMultipleTimes(10,"first")

####################### main program ##########
s = Spectrum()
# s.runMainProgram("VowelCounter_5.py","VowelCounter_GT.py",50000,50,"largest")
# s.produceTestCases(10000,"addingNumbers_GT.py")
s.runMainProgram("Get_Middle_Number_3.py","Get_Middle_Number_GT.py",50000,50,"largest")
# s.runMainProgram("AddingNumbers_1.py","AddingNumbers_GT.py",50000,50,"first")
# s.runMainProgram("StudentMarks_1.py","StudentMarks_GT.py",50000,10,"largest")

# s.visualiseSpectrum()
#s.runMainProgram("UnitConverter.py","UnitConverter_GT.py",10000,10,"largest")
#s.runMainProgram("StudentMarks.py","StudentMarks_GT.py",10000,10,"largest")
#s.visualiseSpectrum()
# s.isMonotonic("VowelCounter.py","VowelCounter_GT.py",50000,10)


####################### run set of experiments ##########
# print(runAllExperiments("experiments.csv",50000,100)) #for final experiment, 50000,100

####################### run monotonic experiments ##########
# print(runAllMonotonicTests([ ["VowelCounter.py","VowelCounter_GT.py"], ["AddingNumbers.py","AddingNumbers_GT.py"] ],50000,10))
# print(runAllMonotonicTests([ ["Get_Middle_Number.py", "Get_Middle_Number_GT.py"] ],50000,10))
                     #for final experiment, 50000,100

####################### visualise an example ##########
#s = Spectrum()
# s.visualiseMonotonicity("VowelCounter.py","VowelCounter_GT.py",50000)
# s.visualiseMonotonicity("Get_Middle_Number_3.py","Get_Middle_Number_GT.py",20000)
# s.visualiseMonotonicity("MultiplyNumbers_1.py","MultiplyNumbers_GT.py",20000)
# s.visualiseMonotonicity("StudentMarks.py","StudentMarks_GT.py",50000)
# s.visualiseMonotonicity("AddingNumbers_1.py","AddingNumbers_GT.py",50000)
