#!/usr/bin/env python
# -- coding:utf-8 --

import time
import os
import re
import math
import Queue
import threading
from support import *

class Segment(object):
    '''
        
    '''
    
    def __init__(self, model, beamSize=10, kMIRA=5, isAllTerminated=False):
        ''' Construct Function
        
        Args:
            model: the learning model for segment
            beamSize: int, search beamSize when decode
            kMIRA: int, kMIRA while use MIRA training
            isAllTerminated: bool, true when is all terminated
        Returns:
            None
        Raise:
            None
        
        '''
        self.model = model
        self.beamSize = beamSize
        self.kMIRA = kMIRA
        self.isAllTerminated = isAllTerminated
    
    def initialize(self, trainSet, numThreads, numPerTheads, threshold=0.0):
        ''' Initialize model
            initialize model args and set up feature mapping
        Args:
            trainSet: SentenceReader, train set in the form of SentenceReader
            numThreads: int, num of threads
            numPerTheads: int, num of train samples per thread
            threshold: float, filter the feature counts less than threshold
        Returns:
            None
        Raise:
            None
        
        '''
        self.model.initialize(trainSet, numThreads, numPerTheads, threshold)
    
    def trainForStructureLinearModel(trainSet, devSet, Iter, miniSize,
        numThreads, trainType, evaluateWhileTraining):
        ''' Main training function
        Args:
            trainSet: SentenceReader, train set in the form of SentenceReader
            devSet: SentenceReader, dev set in the form of SentenceReader
            Iter: count of iteration
            miniSize: int, num of train samples per thread #useless
            numThreads: int, num of threads
            trainType: str, MIRA or Standard
            evaluateWhileTraining: bool, true for eval while training
        Returns:
            None
        Raise:
            None
        '''
        bestAccuracy = 0.0
        bestIter = 0
        bestParams = None
        
        trainSet.reset()
        sentences = []
        while trainSet.hasNext():
            sentences.append(trainSet.next())
        
        # can't understand
        miniSize = 1
        #numThreads = 1
        # number of sentences for each batch
        num = miniSize * numThreads
        # number of batch for each iteration
        batchSize = int(math.ceil(1.0*len(sentences)/num))
        print('Iterate %d times, '
        'the batch size for each iteration is %d'%(Iter, batchSize))
        
        for iter in xrange(Iter):
            print "Iteration %d\n Batch:"%iter
            startTime = time.time()
            random.shuffle(sentences)
            for i in xrange(batchSize):
                start = num * i
                end = start + num
                end = min(end, len(sentences))
                if i%10 == 0:
                    print i,
                
                samples = sentences[start:end]
                actualThreads = min(len(samples), numThreads)
                actualMiniSize = int(math.ceil(1.0*len(samples)/actualThreads))
                
                '''
                multi theads
                '''
                resultQueue = Queue.Queue()
                workers = []
                for k in xrange(actualThreads):
                    curStart = k * actualMiniSize
                    curEnd = min(curStart + actualMiniSize, len(samples))
                    # curEnd2 removed
                    worker = threading.Thread(target=self._trainerTask,
                                    args=(samples, curStart, curEnd,
                                        self, trainType, resultQueue))
                    workers.append(worker)
                
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
                
                ## parse result
                allStatePairs = []
                while not resultQueue.empty():
                    allStatePairs.extend(resultQueue.get())
                
                ## calculate gradient
                gradient = FeatureVector()
                factor = 1.0/len(allStatePairs)
                if trainType == 'MIRA':
                    for states in allStatePairs:
                        K = 0 # number of candidates
                        for kk in xrange(1, len(states)):
                            if states[kk] != None:
                                K += 1
                            else:
                                break
                        b = [0.0 for kk in xrange(K)]
                        lam_dist = [0.0 for kk in xrange(K)]
                        dist = [FeatureVector() for kk in xrange(K)]
                        
                        goldFV = states[0].getGlobalFeatureVector()
                        for kk in xrange(K):
                            # the score difference between 
                            # gold-standard tree and auto tree
                            lam_dist[kk] = (states[0].getScore()
                                        - states[kk+1].getScore())
                            b[kk] = self.loss(states[0], states[kk+1])
                            b[kk] -= lam_dist[kk]
                            #the FV difference
                            dist[k] = FeatureVector.getDistVector(goldFV,
                                        states[k+1].getGlobalFeatureVector())
                        
                        #FIXME@bao: QPSolver.hildreth
                        alpha = QPSolver.hildreth(dist, b)
                        for kk in xrange(K):
                            gradient.add(dist[kk], alpha[kk] * factor)
                    
                else:
                    for states in allStatePairs:
                        if states[1].isGold():
                            continue
                        gradient.add(state[0].getGlobalFeatureVector(), factor)
                        gradient.subtract(state[1].getGlobalFeatureVector(), factor)
                
                avg_upd = 1.0 * Iter * batchSize - (batchSize*(iter-1)+(i+1)) + 1
                self.model.perceptronUpdate(gradient, avg_upd)
            # batch iter end
            endTime = time.time()
            print '\nTrain Time: %f'%(endTime - startTime)
            
            if evaluateWhileTraining:
                startTime = time.time()
                oldParams = self.model.getParam()
                averageParams = self.model.averageParams()
                self.model.setParam(averageParams)
                
                accuracy = self.evaluate(devSet, miniSize, numThreads)
                print 'Dev Acc is %f'$accuracy
                
                if accuracy >= bestAccuracy:
                    bestIter = iter
                    bestAccuracy = accuracy
                    bestParams = averageParams
                
                self.model.setParam(oldParams)
                endTime = time.time()
                print 'Eval time: %f'%(startTime - endTime)
            
            #TODO@bao: eval the gc
            #gc.collect()
        # train iter end
        if bestParams:
            self.model.setParam(bestParams)
        print 'The best iteration is %d'%bestIter

    def decodeBeamSearch(self,sent,trainType):
        '''解码函数
        Args:
            sent:类型为Sentence
            trainType:提供多种模式，有test, standard, early, max,MIRA
        Return:
            一个SegState对象数组，通常为两个元素，MIRA模式下返回多个元素，第一个元素为最佳解，其余为次优解
        Raise:
            None
        '''
        isTrainMode = False
        goldActions = [] # <str>
        goldState = None
        goldActionPosition = 0
        results = [None] * 2
        agenda = []# <SegState>
        heap = []# <SegState>
        scoreBoard = [float("-inf")] * self.beamSize
        # for gold-standard state
        if trainType != "test":
            isTrainMode = True
            goldActions = sent.getAllActions()
            goldState = sent.buildInitState()
        # for max-violation
        if trainType == "max":
            goldPartialStates = []# <SegState>
            predPartialStates = []# <SegState>
            maxViolationPosition = -1
            maxMargin = float("-inf")
        if trainType == "MIRA":
            results = [None] * (self.kMIRA + 1)
        agenda.append(sent.buildInitState())
        circle = 0
        while True:
            circle += 1
            if circle > 1000:
                print "*"
                for state in agenda:
                    print "(%d)"%state.getStep()
            # ==========get gold action for the current step====================
            goldAction = ""
            lengthOfGoldActions = goldActions.__len__()
            if lengthOfGoldActions != 0:
                if goldActionPosition < lengthOfGoldActions:
                    goldAction = goldActions[goldActionPosition]
                goldActionPosition += 1
            if goldAction != "":
                goldState = goldState.transit(goldAction,True,model)
            # ==========one step transit for each state==============
            scoreBoard = [float("-inf")] * self.beamSize
            heap = []
            # build new state
            for state in agenda:
                if state.isTerminated():
                    heap.append(state)
                    continue
                unlabeledFeatures = state.getUnlabeledFeatures()
                actions = self.getAllValidActions(state,isTrainMode)
                for action in actions:
                    labeledFeatures = [] # <str>
                    for feature in unlabeledFeatures:
                        labeledFeatures.append("%s:%s"%(feature,action))
                    score = model.score(model.getFeatureVecotr(labeledFeatures)) + state.getScore()
                    #error handling on variable score
                    if state < min(scoreBoard):
                        continue
                    if goldAction == "":
                        newState = state.transit(action,True,model)
                    else:
                        newState = state.transit(action,goldAction == action,model)
                    if newState.getScore() < min(scoreBoard):
                        continue
                    heap.append(newState)
                    scoreBoard[-1] = newState.getScore()
                    scoreBoard.sort(reverse=True)
            # keep k-best state
            agenda = []
            if heap.__len__() == 0:
                print "Parsing Fault."
                # exit()
            else:
                heap.sort(key=lambda x:x.getScore())
                while (heap.__len__() != 0) and (agenda.__len__() < self.beamSize):
                    agenda.append(heap[-1])
                    del heap[-1]
            # ==========================
            if trainType == "early":
                containedGoldState = None
                for state in agenda:
                    if state.isGold():
                        containedGoldState = state
                        break
                    if containedGoldState == None:
                        results[0] = goldState
                        results[1] = agenda[0]
                        return results
            else:
                if trainType == "max":
                    curMargin = agenda[0].getScore() - goldState.getScore()
                    if curMargin > maxMargin:
                        maxMargin = curMargin
                        maxViolationPosition += 1
                        goldPartialStates.append(goldState)
                        predPartialStates.append(agenda[0])
            # ===========check terminated===================
            if self.isAllTerminated:# terminated when all state in the beam reach terminal state
                allterm = True
                for state in agenda:
                    if not state.isTerminated():
                        allterm = False
                        break
                if allterm:
                    break
            else:# terminated when the best state reach the terminal state
                if agenda.__len__() != 0 and agenda[0].isTerminated():
                    break
        if trainType == "max":
            results[0] = goldPartialStates[maxViolationPosition]
            results[1] = predPartialStates[maxViolationPosition]
        elif trainType == "MIRA":
            results[0] = goldState
            results.extend(agenda)
        else:
            results[0] = goldState
            if agenda.__len__() != 0:
                results[1] = agenda[0]
        return results

    def ParserTask(self,sentences):
        '''解码线程函数
        Args:
            sentences:待解码的Sentence列表
        Return:
            None
        Raise:
            None
        '''
        results = []
        for one in sentences:
            results.append(self.decodeBeamSearch(one,"test")[0].getFinalResult())
        self.wq.put(results)

    def decodeParalle(self,testSet,outpath,numThreads,numPerTheads):
        '''多线程解码函数
        Args:
            testSet:测试数据集
            outpath:解码结果的保存路径
            numThreads:总线程数
            numPerTheads:每个线程中的任务数
        Return:
            None
        Raise:
            None
        '''
        startTime = time.time()
        batch = 0
        testSet.reset()
        while testSet.hasNext():
            print str(batch) + " "
            batch += 1
            # read #numThreads * miniSize instances
            sentences = []
            for i in range(numThreads * numPerTheads):
                if testSet.hasNext():
                    sentences.append(testSet.next())
            LengthOfSentences = sentences.__len__()
            if LengthOfSentences > numThreads:
                actualThreads = numThreads
            else:
                actualThreads = LengthOfSentences
            actualMiniSize = int(math.ceil(LengthOfSentences /float(actualThreads)))
            # wq = Queue.Queue()
            threads = []
            for i in range(actualThreads):
                startPos = actualMiniSize * i
                endPos = startPos + actualMiniSize
                if endPos > LengthOfSentences:
                    endPos = LengthOfSentences
                threads.append(threading.Thread(target=self.ParserTask(sentences[startPos:endPos])))

            # start threads and join main threads
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # fetch the results
            results = []
            while not self.wq.empty():
                results.extend(self.wq.get())
            # write file
            with open(outpath,'w') as outFile:
                for one in results:
                    outFile.write(one + "\n")
        print "Time: %f"%(time.time() - startTime)

    @staticmethod
    def _trainerTask(sentences, start, end, transitioner, trainType, resultQueue):
        ''' parallelising function
        '''
        results = []
        for i in xrange(start, end):
            results.append(transitioner.decodeBeamSearch(sentences[i]), trainType)
        resultQueue.put(results)
    
    def getAllValidActions(self, state, isTrainMode):
        ''' get all valid tag
        Args:
            state: State, the current state
            isTrainMode: bool, isTrainMode
        Returns:
            valideActions: list, contains valid tags
        Raise:
            None
        '''
        preTag = '#'
        if state.getAction():
            preTag = state.getAction()
        valideActions = []
        allPossibleActions = self.model.allPossibleActions()
        for action in allPossibleActions:
            if self._validSequence(preTag, action, isTrainMode):
                valideActions.append(action)
        # valideActions = [action for action in self.model.allPossibleActions()
        #                if self._validSequence(preTag, action, isTrainMode)]
        return valideActions
    
    # validSequence map, used in _validSequence()
    _vsMap = {
        '#':['B', 'S'],
        'B':['I', 'E'],
        'I':['I', 'E'],
        'E':['B', 'S'],
        'S':['B', 'S']
    }
    def _validSequence(self, preTag, tag, isTrainMode):
        ''' used for valid if tag is ok following the preTag
        '''
        if isTrainMode:
            return True
        
        if preTag in self._vsMap:
            if tag in self._vsMap[preTag]:
                return True
            else:
                return False
        else:
            print("Error! In validSequence(), invalid preTag %s"%preTag)
            exit(-1)
        return False
    
    # change to staticmethod maybe better
    def loss(self, goldState, predState):
        ''' Loss function for MIRA training
        '''
        if goldState.getStep() != predState.getStep():
            raise RuntimeError( "Undefined POSTagger.loss()")
        loss_f = 0.0
        while goldState.getAction() and predState.getAction():
            if goldState.getAction() != predState.getAction():
                loss_f += 1.0
            goldState = goldState.getPrevState()
            predState = predState.getPrevState()
        return loss_f
    
    def evaluate(self, devSet, numThreads, numPerTheads):
        ''' evaluate the F_score in the devSet
        '''
        tmpPath = './tmp.seg.%d.%d'%(numThreads, int(time.time()))
        self.decodeParalle(devSet, tmpPath, numThreads, numPerTheads)
        F_score = self.eval(tmpPath, devSet.getPath())
        if os.access(tmpPath, os.F_OK):
            os.remove(tmpPath)
        return F_score
    
    @staticmethod
    def eval(testPath, goldPath):
        ''' evaluate accuracy and return F1 with the given testSet
        '''
        goldReader = SentenceReader(goldPath)
        predReader = SentenceReader(testPath)
        total = 0.0
        corr = 0.0
        while goldReader.hasNext() and predReader.hasNext():
            goldSent = goldReader.next()
            predSent = predReader.next()
            total += goldSent.size()
            for i in xrange(goldSent.size()):
                if goldSent.getAction(i) == predSent.getAction(i):
                    corr += 1.0
        print 'ACC is %f'%(corr)
        
        goldTotal = 0.0
        predTotal = 0.0
        correct = 0.0
        #TODO io handle
        with open(testPath, 'r') as file_in:
            predIn = file_in.readlines()
        with open(goldPath, 'r') as file_in:
            goldIn = file_in.readlines()
        
        for i in xrange(len(predIn)):
            goldSet = self._collect(predIn[i].strip())
            goldTotal += len(goldSet)
            predSet = self._collect(goldIn[i].strip())
            predTotal += len(predSet)
            
            for predSpan in predSet:
                if predSpan in goldSet:
                    correct += 1.0
        
        precision = correct / predTotal
        recall = correct / goldTotal
        F1 = 2 * precision * recall / (precision + recall)
        print 'P=%.2f\tR=%.2f\tF=%.2f'%(precision*100, recall*100, F1*100)
        return F1
    
    @staticmethod
    def _collect(line):
        ''' inner function for eval to collect words
        '''
        wordSet = set()
        words = re.split(r'\s+', line)
        start = 0
        for word in words:
            end = start + len(word)
            wordSet.add((word, start, end))
            start = end
        return wordSet

if __name__ == "__main__":
    seg = Segment(model = None)
    print seg._validSequence(preTag = 'B', tag = 'I', isTrainMode = False)
    print seg._collect('123 456')