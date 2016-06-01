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
            digestList.append(decodedLine["digest"])
            indexList = chidDict.setdefault(decodedLine["chid_1"],list())
            indexList.append(digestIndex)
            digestIndex += 1
    # print chidDict
    # 建立tf-idf模型
    print "tf-idf calculating..."
    # 计算idf值
    myTfIdf = tfidf.TFIDF(digestList)
    # 显示超过10篇中都出现的词及其idf值
    idfDict = myTfIdf.getIdf()
    print [(x.encode('utf-8'),idfDict[x])
        for x in idfDict
        if idfDict[x] < 2
    ]
    tfidfMatrix = {}
    for index,value in chidDict.iteritems():
        # 计算在某一分类中词语的tf值
        tfDict = myTfIdf.getTf(value)
        # 计算TF-IDF值，存在字典中
        for one in myTfIdf.getTfIdf(tfDict,idfDict):
            oneClass = tfidfMatrix.setdefault(one[0],dict())
            oneClass[index] = one[1]
    # save tfidfMatrix
    with open("tf_idf.txt",'w+') as outfile:
        classNames = list(chidDict.keys())
        print >>outfile,"\t%s"%('\t'.join([str(x) for x in classNames]))
        for token,classDict in tfidfMatrix.iteritems():
            List = [token.encode("utf-8")]
            for one in classNames:
                List.append(str(classDict.get(one,0.0)))
            print >>outfile,'\t'.join(List)


if __name__ == '__main__':
    main()