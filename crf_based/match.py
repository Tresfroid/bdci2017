# -*- coding: utf-8 -*-  #
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from data_util import DataUtil

### 创建极性词典
def createPolarityDict():
    # 创建极性词典
    polarDict = {}
    for item in open('dict_add/sentiment_polar.txt').read().strip().split('\n'):
        word = item.strip().split('\t')[0].decode('utf-8')
        polarity = item.strip().split('\t')[1]
        polarDict[word] = polarity

    # for item in open('dict_add/polarity_dict_add.csv').read().strip().split('\n'):
    #     if item == 'Instance,po':
    #         continue
    #     word = item.strip().split(',')[0].decode('utf-8')
    #     polarity = item.strip().split(',')[1]
    #     polarDict[word] = polarity

    # for item in open('needPolarity1.csv').read().strip().split('\n'):
    #     word = item.strip().split(',')[0].decode('utf-8')
    #     polarity = item.strip().split(',')[1]
    #     polarDict[word] = polarity

    return polarDict

# ###从CRF输出文件中提取出每个句子对应的标注序列
def getTagSequenceList(file, separator):
    # 从CRF输出文件中提取出每个句子对应的标注序列
    f = open(file, 'r')
    sentencesList = f.read().strip().split('\n\n')
    f.close()
    tagSequenceList = []

    for sentence in sentencesList:
        tagSequence = ""
        words = sentence.split('\n')

        for word in words:
            tag = word.strip().split(separator)[-1]
            tagSequence += tag

        tagSequenceList.append(tagSequence)

    return tagSequenceList


#### 通过标注序列提取出配对的主题词与情感词的索引对
# def getPair(tagSequence):
#     # 通过标注序列提取出配对的主题词与情感词的索引对
#     pairList = []
#     preWord = {'tag':'null', 'index': -1}
#
#     for index in range(len(tagSequence)):
#         if tagSequence[index] == 'T' and preWord['tag'] == 'null':
#             preWord['tag'] = 'T'
#             preWord['index'] = index
#         elif tagSequence[index] == 'S' and preWord['tag'] == 'null':
#             preWord['tag'] = 'S'
#             preWord['index'] = index
#             pairList.append([-1, index])
#         elif tagSequence[index] == 'T' and preWord['tag'] == 'T':
#             preWord['tag'] = 'T'
#             preWord['index'] = index
#         elif tagSequence[index] == 'S' and preWord['tag'] == 'T':
#             pairList.append([preWord['index'], index])
#             preWord['tag'] = 'S'
#             preWord['index'] = index
#         elif tagSequence[index] == 'T' and preWord['tag'] == 'S':
#             preWord['tag'] = 'T'
#             preWord['index'] = index
#         elif tagSequence[index] == 'S' and preWord['tag'] == 'S':
#             preWord['tag'] = 'S'
#             preWord['index'] = index
#             pairList.append([-1, index])
#
#
#
#     return pairList

# ###带分句的标注序列配对
def getPairMod(tagSequence):
    # 带分句的标注序列配对
    pairList = []
    preWord = {'tag': 'null', 'index': -1}

    if type(tagSequence) == float:
        return pairList

    for index in range(len(tagSequence)):
        if tagSequence[index] == ',':
            preWord = {'tag': 'null', 'index': -1}
            continue
        if tagSequence[index] == 'T' and preWord['tag'] == 'null':
            preWord['tag'] = 'T'
            preWord['index'] = index
        elif tagSequence[index] == 'S' and preWord['tag'] == 'null':
            preWord['tag'] = 'S'
            preWord['index'] = index
            pairList.append([-1, index])
        elif tagSequence[index] == 'T' and preWord['tag'] == 'T':
            preWord['tag'] = 'T'
            preWord['index'] = index
        elif tagSequence[index] == 'S' and preWord['tag'] == 'T':
            pairList.append([preWord['index'], index])
            preWord['tag'] = 'S'
            preWord['index'] = index
        elif tagSequence[index] == 'T' and preWord['tag'] == 'S':
            preWord['tag'] = 'T'
            preWord['index'] = index
        elif tagSequence[index] == 'S' and preWord['tag'] == 'S':
            preWord['tag'] = 'S'
            preWord['index'] = index
            pairList.append([-1, index])

    segement = tagSequence.split(',')
    for i in range(len(segement)):
        if segement[i].replace('O', '') == 'ST':
            indexS = segement[i].index('S')
            indexT = segement[i].index('T')
            preLength = 0
            for j in range(i):
                preLength += len(segement[j])+1
                # print len(segement[j])
            indexS += preLength
            indexT += preLength
            # print indexS, indexT
            pairList.remove([-1, indexS])
            pairList.append([indexT, indexS])

    return pairList

def writePairToData(readFile, writeFile, pair):
    # 通过readFile读取分词信息，将配对的索引信息对转化为配对的词对
    # 将主题词、情感词、情感极性写入到writeFile文件里面
    polarDict = createPolarityDict()
    data_util = DataUtil()
    data = data_util.load_data1(readFile)
    themeList = []
    sentiment_wordList = []
    sentiment_anlsList = []
    notInPolarDict = []

    for i in range(len(data)):
        # 第i行数据
        themeStr = ''
        sentiment_wordStr = ''
        sentiment_anlsStr = ''
        if type(data.loc[i, 'new_words']) != float:

            words = data.loc[i, 'new_words'].strip().split(' ')

            if len(pair[i]) != 0:
                for index1, index2 in pair[i]:
                    # print len(words), index1, index2
                    if index1 == -1:
                        themeStr += 'NULL;'
                    else:
                        # if index1 + 1 > len(words):
                        #     print 'index1', i
                        themeStr += words[index1]+';'

                    # if index2 + 1 > len(words):
                    #     print 'index2', i;
                    #     continue
                    sentiment_wordStr += words[index2]+';'

                    if polarDict.has_key(words[index2]):
                        sentiment_anlsStr += polarDict[words[index2]] + ';'
                    else:
                        if words[index2] not in notInPolarDict:
                            notInPolarDict.append(words[index2])
                            # print words[index2]
                        sentiment_anlsStr += '-1;'

                    # if words[index2] == u'渣脚':
                    #     sentiment_anlsStr += '-1;'
                    # else:
                    #     sentiment_anlsStr += polarDict[words[index2]] + ';'

        themeList.append(themeStr)
        sentiment_wordList.append(sentiment_wordStr)
        sentiment_anlsList.append(sentiment_anlsStr)

    # f = open('needPolarity3.csv', 'w')
    # for word in notInPolarDict:
    #     f.write(word+',\n')
    # f.close()
    # print len(notInPolarDict)
    data["theme"] = themeList
    data["sentiment_word"] = sentiment_wordList
    data["sentiment_anls"] = sentiment_anlsList

    data_util.save_data(data, writeFile)

def testPairPrecision():
    data_util = DataUtil()
    data = data_util.load_data1('result2.csv')
    n = 0
    for i in range(len(data)):
        theme1 = data.loc[i, 'theme']
        theme2 = data.loc[i, 'theme2']

        sentiment_word1 = data.loc[i, 'sentiment_word']
        sentiment_word2 = data.loc[i, 'sentiment_word2']
        if (type(theme1) == type(theme2) and set(theme1.split(';')) == set((theme2).split(';'))
            and type(sentiment_word1) == type(sentiment_word2) and set(sentiment_word1.split(';')) == set((sentiment_word2).split(';'))):
            n += 1

    print float(n) / len(data)

def combineNegativeWordsAndSentimentWords(inputFile, outputFile):
    data_util = DataUtil()
    data = data_util.load_data1(inputFile)
    negativeWordList = ['不', '不是', '不是很', '不够',"没","没有"]
    wordsList = []
    labelList = []
    for i in range(len(data)):
        wordsStr = ''
        labelStr = ''
        if type(data.loc[i, 'new_words']) != float:
            words = data.loc[i, 'new_words'].strip(',').strip().split(' ')
            labels = list(data.loc[i, 'label'].strip())
            if len(words) < len(labels):
                print '第',i+1,'行words长度为', len(words),'labels长度为',len(labels)
            for j in range(len(words)):
                if j+1 < len(words):
                    if words[j] in negativeWordList and labels[j+1] == 'S':
                        words[j] = words[j] + words[j+1]
                        # print i,words[j]
                        del words[j+1]
                        del labels[j]

            for k in range(len(words)):
                wordsStr += words[k]+' '
                labelStr += labels[k]
        wordsList.append(wordsStr.strip())
        labelList.append(labelStr.strip())
    data['words'] = wordsList
    data['label'] = labelList
    data_util.save_data(data, outputFile)

if __name__ == '__main__':

    # combineNegativeWordsAndSentimentWords('data/test_fc_labe2.csv', 'data/result2.csv')

    # 获得词性标注序列
    # tagSL = getTagSequenceList("data/output_test.txt")
    # data_util = DataUtil()
    # data = data_util.load_data1('data/result2.csv')
    # tagSL = []
    # for i in range(len(data)):
    #     tagSq = data.loc[i, 'label']
    #     tagSL.append(tagSq)
    #
    # # 获得所有句子的配对信息
    # pair = []
    # for i in range(len(tagSL)):
    #     pairList = getPairMod(tagSL[i])
    #     # if i == 442:
    #     #     print pairList
    #     pair.append(pairList)
    #     # print pairList
    #
    # # 通过配对信息、分词信息，得到最终输出文件
    # writePairToData("data/result2.csv", "data/result2.csv", pair)
    # testPairPrecision()

    tagSL = getTagSequenceList("data/output_test.txt",'\t')
    # tagSL = getTagSequenceList("data/output.txt")

    # 获得所有句子的配对信息
    pair = []
    for tagS in tagSL:
        pairList = getPairMod(tagS)
        pair.append(pairList)

    # 通过配对信息、分词信息，得到最终输出文件
    writePairToData("data/test_data_parents_feature.csv", "data/result.csv", pair)




