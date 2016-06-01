#encoding=utf-8
import tfidf
import json
import sys

'''
得到总文章集，以及各类的分布
'''
def getTotalList(filename,keyword = "digest",chid = "chid_1"):
    # 用于存储所有的摘要字段
    totalList = []
    # 用于对类别进行标记
    chidDict = {}
    index = 0
    # 开始解析json
    print "json parsing..."
    with open(filename,'r') as infile:
        for line in infile.xreadlines():
            decodedLine = json.loads(line)
            # 没有类别无法计算各个指标，跳过
            if "chid_1" not in decodedLine:
                continue
            totalList.append(decodedLine.get(keyword,""))
            indexList = chidDict.setdefault(decodedLine[chid],list())
            indexList.append(index)
            index += 1
    return totalList,chidDict

def calcTfIdf(totalList,chidDict,keyword = "digest"):
    # 建立tf-idf模型
    print "tf-idf calculating..."
    myTfIdf = tfidf.TFIDF(totalList)
    # 计算idf值
    idfDict = myTfIdf.getIdf()
    # 显示超过总文章数目的1%中都出现的词及其idf值
    print [(x.encode('utf-8'),idfDict[x])
        for x in idfDict
        if idfDict[x] < 2
    ]
    # 计算在某一分类中词语的tf值
    tfidfMatrix = {}
    for index,value in chidDict.iteritems():
        tfDict = myTfIdf.getTf(value)
        # 计算TF-IDF值，存在tfidfMatrix中
        for one in myTfIdf.getTfIdf(tfDict,idfDict):
            oneClass = tfidfMatrix.setdefault(one[0],dict())
            oneClass[index] = one[1]
    # save tfidfMatrix
    with open("%s_tfidf.txt"%keyword,'w+') as outfile:
        classNames = list(chidDict.keys())
        print >>outfile,"\t%s"%('\t'.join([str(x) for x in classNames]))
        for token,classDict in tfidfMatrix.iteritems():
            List = [token.encode("utf-8")]
            for one in classNames:
                List.append(str(classDict.get(one,0.0)))
            print >>outfile,'\t'.join(List)

def main():
    # totalList是总文件集，chidDict是各个类别的文章分布
    keyword = "digest"
    totalList,chidDict = getTotalList("dirtyData.dat",keyword)
    calcTfIdf(totalList,chidDict)



if __name__ == '__main__':
    main()