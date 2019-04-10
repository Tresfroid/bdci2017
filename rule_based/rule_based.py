#!/user/bin/python
# -*- coding: utf-8 -*-   Python只检查#、coding和编码字符串，其他的字符都是为了美观加上的
import sys
import os
import codecs
import pandas as pd
from data_util import DataUtil
# reload(sys)
# sys.setdefaultencoding('utf-8')



def match(inputFile,outputFile):
    theme_list = []###主题词典
    sentiment_list= []###情感词典
    # theme_dict = open("dict_add/theme_set.txt",'r')
    # sentiment_dict = open('dict_add/sentiment_set.txt', 'r')
    with codecs.open('dict_add/theme_set.txt', encoding='utf-8') as f:
        theme_list = [line.strip() for line in f]
    with codecs.open('dict_add/sentiment_set.txt', encoding='utf-8') as f:
        sentiment_list = [line.strip() for line in f]
    # for line in theme_dict:
    #     theme = line.strip()
    #     theme_list.append(theme)
    # for line in sentiment_dict:
    #     sentiment = line.strip()
    #     sentiment_list.append(sentiment)
    # print theme_list
    # print sentiment_list
    data_util = DataUtil()
    data = data_util.load_data1(inputFile)
    label_list = []
    for i in range(len(data)):
        theme_all_list = []
        sentiment_all_list = []
        temp=str(data.loc[i,'new_words'])
        if temp[-1] == ",":
#             print  temp
            temp = temp[:-1]
        elif temp[0]==",":
            temp = temp[1:]
        words_douhao = temp.strip(",").split(" , ")


        # print "".join(words_douhao)
        # print data.loc[i,'words']
        # print type(data.loc[i,'words'])
        label = ""
        for words in words_douhao:

            theme_result = []###找到的主题
            sentiment_result = []###找到的情感
            word = words.strip(" ").split(" ")

            for w in word:
                if w in theme_list:
                    theme_result.append(w)
                    label+="T"
                elif (w in sentiment_list):
                    sentiment_result.append(w)
                    label+="S"
                else:
                    label+="O"
            label+=","
        label_list.append(label[:-1])
    data["label"] =label_list
    data_util.save_data(data,outputFile)



            # if len(theme_result) == len(sentiment_result):
            #     for t in theme_result:
            #         theme_all_list.append(t)
            #     for s in sentiment_result:
            #         sentiment_all_list.append(s)
            # elif len(theme_result)<len(sentiment_result):
            #     for i in range(len(theme_result)):
            #         theme = theme_list[i]
            #         sentiment = sentiment_list[]
            # elif len(theme_result) > len(sentiment_result):
            #     for i in range(len(sentiment_result)):
            #         theme_all_list.append(theme_result[])


def pro():
    train_data_file_path = "data/data2.csv"
    data_util = DataUtil()
    data = data_util.load_data1(train_data_file_path)

    train_data_file_path2 = "data/sentiment_pro.csv"
    train_data_file_path3 = "data/theme.csv"
    data2 = data_util.load_data1(train_data_file_path2)
    data3 = data_util.load_data1(train_data_file_path3)
    # data_util = DataUtil()
    # # data1 = data_util.load_data01(train_data_file_path1)
    # # data2 = data_util.load_data01(train_data_file_path)
    # # data3 = data_util.load_data01(train_data_file_path)
    # data1=pd.read_csv(train_data_file_path1,sep=",",encoding="utf8",names=[u"row_id",u"words"],index_col=None)
    # data2 = pd.read_csv(train_data_file_path2, sep=" ", encoding="utf8", names=[u"sentiment_word"])
    # data3 = pd.read_csv(train_data_file_path3, sep=" ", encoding="utf8", names=[u"theme"])
    # data1["theme"] = data3["theme"]
    # data1["sentiment_word"]=data2["sentiment_word"]
    # data["words_new"]=list
    # data_util.save_data(data,"data/data1.csv")
    # list = []
    # for i in range(len(data)):
    #     sentence = data.loc[i, "words"]
    #     sentence = re.sub(u"\?\?", "", sentence)
    #     sentence = re.sub(u"\?+", ",", sentence)
    #     sentence = re.sub(u"\!+", ",", sentence)
    #     sentence = re.sub(u"\.+", ",", sentence)
    #     sentence = re.sub(u"，+", ",", sentence)
    #     sentence = re.sub(u"？+", ",", sentence)
    #     sentence = re.sub(u"！+", ",", sentence)
    #     sentence = re.sub(u"。+", ",", sentence)
    #     sentence = re.sub(u"；+", ",", sentence)
    #     sentence = re.sub(u";+", ",", sentence)
    #     sentence = re.sub(u",+", ",", sentence)
    #     sen = sentence
    #     sen = re.sub("(,\s)+", ", ", sen)
    #     sen1 = re.sub("\s+", " ", sen)
    #     sent = sen1.strip(" ")
    #     sent = sent.strip(",")
    #     sent = sent.strip(" ")
    #     sent = re.sub(" , , ", " , ", sent)
    #     sent = re.sub(" , ,", " , ", sent)
    #     sent = sent.strip(",")
    #     sent = sent.strip(" ")
    #     list.append(sent)
    # data["new_words"] = list
    # data_util.save_data(data, "data/data2.csv")
    # data2 = pd.read_csv(train_data_file_path2, sep=" ", encoding="utf8", names=[u"sentiment_word"])
    # data3 = pd.read_csv(train_data_file_path3, sep=" ", encoding="utf8", names=[u"theme"])
    # data_util.save_data(data2,"data/sentiment_pro.csv")
    # data_util.save_data(data3,"data/theme.csv")
    for i in range(len(data)):
        sents = str(data2[i, "sentiment_word"]).strip(" ").strip(";")[:-1]
        themes= str(data3[i, "theme"]).strip(" ").strip(";")[:-1]
        clause= str(data[i, "new_words"]).strip(" ").strip(",")

if __name__ == '__main__':
    match("data/test_fc_data.csv","data/test_fc_label.csv")
    # pro()