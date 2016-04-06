class Sentence(object):
    def getAllActions(self):
        return []
    def buildInitState(self):
        return SegState()

class SegState(object):
    def getStep(self):
        return 0
    def transit(self,action,isGold,model):
        return SegState()
    def isTerminated(self):
        return False
    def getUnlabeledFeatures(self):
        return []
    def getScore(self):
        return 0
    def isGold(self):
        return False
    def getAction(self):
        return ""
    def getFinalResult(self):
        return "1"

class Model(object):
    def score(self,fv):
        return 0
    def getFeatureVecotr(self,features):
        return []
    def allPossibleActions(self):
        return []
    
class SentenceReader(object):
    """docstring for SentenceReader"""
    def __init__(self):
        super(SentenceReader, self).__init__()
    def reset(self):
        return
    def hasNext(self):
        return True
    def next(self):
        return Sentence()

class FeatureVector(object):
    """docstring for FeatureVector"""
    @staticmethod
    def dotProduct(fv1,fv2):
        s1 = set(fv1.keys())
        s2 = set(fv2.keys())
        result = 0
        for key in s1&s2:
            result += fv1[key] * fv2[key]
        return result