import math

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

class QPSolver(object):
    """
    docstring for QPSolver
    """
    MAX_ITER = 10000
    EPS = 1e-8
    ZERO = 1e-16
    def hildreth(self, fv, label):
        LengthOfb = label.__len__()
        alpha = [0.0] * LengthOfb
        F = [0.0] * LengthOfb
        kkt = [0.0] * LengthOfb
        K = fv.__len__()
        GramMatrix = [[0]*K for i in range(K)]
        is_computed = [False] * K
        for i in range(K):
            GramMatrix[i][i] = FeatureVector().dotProduct(fv[i],fv[i])
        #find maximum kkt = F = label
        max_kkt = float("-inf")
        max_kkt_i = -1
        for i in range(LengthOfb):
            F[i] = label[i]
            kkt[i] = F[i]
            if kkt[i] > max_kkt:
                max_kkt = kkt[i]
                max_kkt_i = i
        circle = 0
        diff_alpha = 0.0
        try_alpha = 0.0
        add_alpha = 0.0
        while max_kkt >= self.EPS and circle < self.MAX_ITER:
            diff_alpha = F[max_kkt_i] / GramMatrix[max_kkt_i][max_kkt_i]
            if GramMatrix[max_kkt_i][max_kkt_i] <= self.ZERO:
                diff_alpha = 0.0
            try_alpha = alpha[max_kkt_i] + diff_alpha
            if try_alpha < 0.0:
                add_alpha = -1.0 * alpha[max_kkt_i]
            else:
                add_alpha = diff_alpha
            alpha[max_kkt_i] += add_alpha
            if not is_computed[max_kkt_i]:
                for i in range(K):
                    GramMatrix[i][max_kkt_i] = FeatureVector().dotProduct(fv[i],fv[max_kkt_i])
                    is_computed[max_kkt_i] = True
            for i in range(LengthOfb):
                F[i] -= add_alpha * GramMatrix[i][max_kkt_i]
                kkt[i] = F[i]
                if alpha[i] > self.ZERO:
                    kkt[i] = abs(F[i])
            #find maximum kkt
            max_kkt = float("-inf")
            max_kkt_i = -1
            for i in range(LengthOfb):
                if kkt[i] > max_kkt:
                    max_kkt = kkt[i]
                    max_kkt_i = i
            circle += 1
        return alpha

def main():
    qp = QPSolver()
    print qp.hildreth([{1:1.0,2:1.0,3:1.0},{2:1.0,3:1.0,6:1.0}],[1,-1])
if __name__ == '__main__':
    main()