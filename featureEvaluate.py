#encoding=utf-8
import tfidf
import json
import sys

def main():
    # 用于存储所有的摘要字段
    digestList = list()
    digestIndex = 0
    # 用于对类别进行标记
    chidDict = dict()
    # 开始解析json
    print "json parsing..."
    with open("dirtyData.dat",'r') as infile:
        for line in infile.xreadlines():
            decodedLine = json.loads(line)
            digestList.append(decodedLine["digest"].encode("utf-8"))
            indexList = chidDict.setdefault(decodedLine["chid_1"],list())
            indexList.append(digestIndex)
            digestIndex += 1
    # 建立tf-idf模型
    print "tf-idf calculating..."
    myTfIdf = tfidf.TFIDF(digestList)
    # 计算idf值
    idfDict = myTfIdf.getIdf()
    tfidfMatrix = {}
    for index,value in chidDict.iteritems():
        # 计算在某一分类中词语的tf值
        tfDict = myTfIdf.getTf(value)
        # 计算TF-IDF值，存在字典中
        for one in myTfIdf.getTfIdf(tfDict,idfDict):
            oneClass = tfidfMatrix.setdefault(one[0],dict())
            oneClass[index] = one[1]
    # print tfidfMatrix
    with open("tf_idf.csv",'w+') as outfile:
        classNames = list(chidDict.keys())
        print >>outfile,",%s"%(','.join([str(x) for x in classNames]))
        for token,classDict in tfidfMatrix.iteritems():
            List = [token]
            for one in classNames:
                List.append(str(classDict.get(one,0.0)))
            print >>outfile,','.join(List)


if __name__ == '__main__':
    main()