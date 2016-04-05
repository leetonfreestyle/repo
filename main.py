from support import * 
import math
import Queue
import threading
import time


class Segmenter(object):
    kMIRA = 5
    beamSize = 10
    model = Model()
    wq = Queue.Queue()
    isAllTerminated = False
    # validSequence map, used in _validSequence()
    _vsMap = {
        '#':['B', 'S'],
        'B':['I', 'E'],
        'I':['I', 'E'],
        'E':['B', 'S'],
        'S':['B', 'S']
    }

    def _validSequence(self, preTag, tag, isTrainMode):
        '''
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

    def getAllValidActions(self, state, isTrainMode):
        '''
        '''
        preTag = '#'
        if state.getAction():
            preTag = state.getAction()
        valideActions = []
        allPossibleActions = self.model.allPossibleActions()
        for action in allPossibleActions:
            if self._validSequence(preTag, action, isTrainMode):
                valideActions.add(action)
        return valideActions

    def decodeBeamSearch(self,sent,trainType):
        '''
        '''
        isTrainMode = False
        goldActions = []
        goldState = None
        goldActionPosition = 0
        results = [None] * 2
        agenda = []#size = beamSize
        heap = []#size = beamSize
        scoreBoard = [float("-inf")] * self.beamSize
        # for gold-standard state
        if trainType != "test":
            isTrainMode = True
            goldActions = sent.getAllActions()
            goldState = sent.buildInitState()
        # for max-violation
        if trainType == "max":
            goldPartialStates = []#ArrayList<SegState>
            predPartialStates = []#ArrayList<SegState>
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
                    labeledFeatures = []#size
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
        '''
        '''
        results = []
        for one in sentences:
            results.append(self.decodeBeamSearch(one,"standard")[0].getFinalResult())
        self.wq.put(results)

    def decodeParalle(self,testSet,outpath,numThreads,numPerTheads):
        '''
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
            break
        print "Time: %f"%(time.time() - startTime)

def main():
    sg = Segmenter()
    # sg.decodeBeamSearch(Sentence(),'standard')
    sg.decodeParalle(SentenceReader(),"test.txt",2,1)
if __name__ == '__main__':
    main()
