# -*- coding: utf-8 -*-  #

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import pandas as pd
import logging
import  re
from pyltp import Segmentor,Postagger,Parser
import jieba 
jieba.load_userdict("dict_add/word4seg.txt")
# import gensim
# model = gensim.models.Word2Vec.load("D:/word2vec/Word60.model")
# from cryptography.hazmat.primitives.serialization import Encoding

segmentor = Segmentor()
segmentor.load('LTP\ltp_data\cws.model')
postagger = Postagger()
postagger.load("LTP\ltp_data\pos.model")
parser = Parser()
parser.load("LTP\ltp_data\parser.model")

class DataUtil(object):

    def load_data(self,path,header=True):
        """
        读取数据
        :param path: 数据文件路径
        :return:
        """
        """header:指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。
            header为true则文件没有列名，false为文件有列名
        """
        if header:
            data = pd.read_excel(path,
                               sep="\t",
                               header=0,
                               encoding="utf8")
        else:
            data = pd.read_excel(path,
                               sep="\t",
                               header=None,
                               encoding="utf8")
        return data

    def load_data1(self,path,header=True):
        """
        读取数据
        :param path: 数据文件路径
        :return:
        """
        """header:指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。
            header为true则文件没有列名，false为文件有列名
        """
        if header:
            data = pd.read_csv(path,
                               sep=",",
                               header=0,
                               encoding="utf8")
        else:
            data = pd.read_csv(path,
                               sep=",",
                               header=None,
                               encoding="utf8")
        return data

    def save_data(self,data,path):
        """
        保存数据
        :param path:数据文件的路径
        :return:
        """
        data.to_csv(path,
                    sep=",",
                    header=True,
                    index=False,
                    encoding="utf8")

    def processing_na_value(self,data,clear_na=True,fill_na=False,fill_char="NONE_NULL",columns=None):

        logging.debug('[def processing_na_value()] 对缺失值进行处理....')
        for column in data.columns:
            if columns == None or column in columns:
                data[column] = data[column].replace(r"^\s*$",np.nan,regex=True)
                count_null =sum(data[column].isnull())
                if count_null!=0:
                    logging.warn(u'%s字段有空值，个数：%d' % (column, count_null))
                    if clear_na:
                        logging.warn(u'对数据的%s字段空值进行摘除'%(column))
                        data = data[data[column].notnull()].copy()
                    else:
                        if fill_na:
                            logging.warn(u'对数据的%s字段空值进行填充，填充字符为：%s'%(column,fill_char))
                            data[column] = data[column].fillna(value=fill_char)

        return data

    def segment_sentence(self,sentence):
        sentence = re.sub(u"\?\?", "", sentence)
        sentence = re.sub(u"\?+", ",", sentence)
        sentence = re.sub(u"\!+", ",", sentence)
        sentence = re.sub(u"\.+", ",", sentence)
        sentence = re.sub(u"，+", ",", sentence)
        sentence = re.sub(u"？+", ",", sentence)
        sentence = re.sub(u"！+", ",", sentence)
        sentence = re.sub(u"。+", ",", sentence)
        sentence = re.sub(u"\s+", "", sentence)
        sentence = re.sub(u",+", ",", sentence)
        sentence = re.sub(u"~+", "", sentence)
        sentence = re.sub(u"@+", "", sentence)
        sentence = re.sub(u"(\<br\/\>)+", "", sentence)
        sentence = re.sub(u"(\<\/br\>)+", "", sentence)
        sentence = re.sub(u"\[追评\]", "", sentence)
        sentence = re.sub(u"\[ 追评 \]", "", sentence)
        sentence1 = sentence.split(",")#
        sen=" "
        for item in sentence1:
            clause=jieba.cut(item.strip())
            temp=""
            for i in clause:
                if i in ["很","太","再","【","】","\"","、","：","；","；","（","）","「","」","〖","〗","』","『","#","$","%","*","&","⊙","(",")","/","～","^","…"," 、 "]:
                    i=""
                else:
                    temp+="%s "%(i)
            sen +=temp.strip(" ")
            sen=sen+" , "
        sen = re.sub("(,\s)+", ", ", sen)
        sen1 = re.sub("\s+", " ", sen)
        sent=sen1.strip(" ")
        sent=sent.strip(",")
        sent=sent.strip(" ")
        sent=re.sub(" , , ", " , " , sent)
        sent = re.sub(" , ,", " , ", sent)
        sent=sent.strip(",")
        sent=sent.strip(" ")

        return sent


    # 将数据随机切割成训练集和测试集
    def split_train_test(self,data, train_split=0.8):

        logging.debug('对数据随机切分成train和test数据集，比例为：%f' % (train_split))
        num_train = len(data)
        num_dev = int(num_train * train_split)
        num_test = num_train - num_dev
        logging.debug('全部数据、训练数据和测试数据的个数分别为：%d,%d,%d' % (num_train, num_dev, num_test))
        rand_list = np.random.RandomState(0).permutation(num_train)
        # print rand_list
        # print rand_list[:num_dev]
        # print rand_list[num_dev:]
        dev_data = data.iloc[rand_list[:num_dev]].sort_index()
        test_data = data.iloc[rand_list[num_dev:]].sort_index()
        # print dev_data
        # print test_data
        return dev_data, test_data



##训练数据预处理数据+分词
def preprocess_train_data(inputFile,outputFile):

    data_util = DataUtil()
    
    data = data_util.load_data(inputFile)

       
    # -------------- region start : 1. 处理空值数据 -------------
    logging.debug('-' * 20)
    print '-' * 20

    logging.debug('1. 处理空值数据')
    print '1. 处理空值数据'

    print '原始数据有%d条句子'%(len(data))
    # 将TEXT字段有空数值的数据去除，并保存
    data = data_util.processing_na_value(data, clear_na=True,columns=[u'sentiment_anls'])
    print '去除sentiment_anls字段有空数值的数据之后，剩下%d条句子'%(len(data))

    logging.debug('-' * 20)
    print '-' * 20
    # -------------- region end : 1. 处理空值数据 ---------------
    print ('分词')    
    data['words'] = data['content'].apply(data_util.segment_sentence)

    print ('保存数据')
    # 保存数据
    # data_util.save_data(data, 'data/train_data_0_%d.csv'% len(data))
    
    # 将其他字段有空数值的数据去除
    data = data_util.processing_na_value(data, clear_na=True)
    # output_file_path = 'data/train_data_1.csv'
    print '去除其他字段有空数值的数据之后，剩下%d条句子,输出到：%s'%(len(data),outputFile)
    # 保存数据
    data_util.save_data(data, outputFile)


##词性、依存句法分析
def pos_and_par(inputFile,outputFile):
    # train_data_file_path ="data/train_data_1.csv"
    # train_data_file_path ="data/test_fc_data.csv"
    # train_data_file_path = "data/test_data_3531.csv"

    data_util = DataUtil()
    data = data_util.load_data1(inputFile)
    pos_list = []
    par_list = []
    for i in range(len(data)):
        # print type(data.loc[i,'words'])
        id = data.loc[i,'row_id']
        # words_ori = data.loc[i,'new_words'].split(" ")
        # if(type(data.loc[i,'new_words'])==float):
        #     pos_list = ['n ']
        #     par_list = ['0:HED']
        #     print "1"
        # else:
        words_ori = data.loc[i,'words'].strip().split(" ")

        # for i in words_ori:
        #     print  i
        # print words_ori
        list1=[]
        for i in words_ori:
            print id
            # print i.encode("utf-8")
            list1.append(i.encode("utf-8"))

        # print ('词性标注')
        pos = postagger.postag(list1)
        # print ' '.join(pos)

        # print ('句法分析')
        pars = parser.parse(list1,pos)
        # print ' '.join("\t".join("%d:%s" % (par.head, par.relation) for par in pars))

        #将词性list转为字符串
        str=""
        for it in pos:
            str+="%s " % it

        str1=""
        for par in pars:
            str1+="%s:%s " % (par.head,par.relation)
        pos_list.append(str)
        par_list.append(str1)

    data["pos"]=pos_list
    data["parser"]=par_list
    data_util.save_data(data, outputFile)
    # data_util.save_data(data,"data/test_data_pos_pars.csv")
    # data_util.save_data(data,"data/train_pos_pars.csv")


##加入父亲词特征
def parents_feature(inputFile,outputFile):
    # inputfile_path = "data/test_pos_pars.csv"
    # inputfile_path = "data/train_pos_pars.csv"
    # inputfile_path = "data/test_data_pos_pars.csv"
    data_util = DataUtil()
    data = data_util.load_data1(inputFile)
    parents_word_total_list = []
    parents_pos_total_list = []
    parents_parser_total_list = []
    for i in range(len(data)):
        words_list = []  #存放该句分词结果
        pos_list = []   #存放该句词性标注结果
        parents_index_list = []  #存放父亲词index
        parents_word_list = []  #存放父亲词的词本身
        parents_pos_list = []  #存放父亲词词性
        parents_parser_list = []  #存放当前词与父亲词依存关系
        id = data.loc[i,"row_id"]
        words = data.loc[i,"words"].strip(" ").split(" ")
        pos = str(data.loc[i,"pos"]).strip(" ").split(" ")
        parser = str(data.loc[i, "parser"]).strip(" ").split(" ")
        for w in words:
            words_list.append(w.encode("utf-8"))
        # print " ".join(words_list)
        for p in pos:
            pos_list.append(p.encode("utf-8"))
        for p in parser:
            parents_index_list.append(p.split(":")[0].encode("utf-8"))
            parents_parser_list.append(p.split(":")[1].encode("utf-8"))
        print id
        print " ".join(parents_index_list)

        for i in parents_index_list:
            index = int(i)-1
            if (index==-1):
                parents_word_list.append("NULL")
                parents_pos_list.append("NULL")
            else:
                parents_word_list.append(words_list[index])
                parents_pos_list.append(pos_list[index])

        parents_word_str = ""
        for i in parents_word_list:
            parents_word_str += "%s " % i
        parents_word_total_list.append(parents_word_str)

        parents_pos_str = ""
        for i in parents_pos_list:
            parents_pos_str += "%s "%i
        parents_pos_total_list.append(parents_pos_str)

        parents_parser_str = ""
        for i in parents_parser_list:
            parents_parser_str +="%s "%i
        parents_parser_total_list.append(parents_parser_str)

    data["parents_word"] = parents_word_total_list
    data["parents_pos"] = parents_pos_total_list
    data["parents_parser"] = parents_parser_total_list

    # data_util.save_data(data, "train_test/test_data_3531_pos_pars_parents_feature.csv")
    # data_util.save_data(data, "data/train_parents_feature.csv")
    # data_util.save_data(data, "data/test_data_parents_feature.csv")
    data_util.save_data(data, outputFile)
        # print " ".join(parents_word_list)
        # print " ".join(parents_pos_list)
        # print " ".join(parents_parser_list)


###加入左右词特征
def get_word_left_right(inputFile,outputFile):
    left = []
    right = []
    left_pos = []
    right_pos = []
    # train_data_file_path="data/train_parents_feature.csv"
    # train_data_file_path="data/test_data_parents_feature.csv"
    # train_data_file_path = "result_train_test/test_data_3531_pos_pars_parents_feature.csv"
    data_util = DataUtil()
    data = data_util.load_data1(inputFile)

    for i in range(len(data)):
        words = data.loc[i, "words"].split(" ")
        pos = data.loc[i, "pos"].split(" ")[:-1]
        left_word = ""
        left_word_pos = ""
        right_word_pos = ""
        right_word = ""
        for j in range(len(words)):
            if (j == 0):
                left_word += "NULL "
                left_word_pos += "NULL "
            else:
                left_word += "%s " % words[j - 1]
                left_word_pos += "%s " % pos[j - 1]

        for k in range(len(words)):
            if k == len(words) - 1:
                right_word += "NULL "
                right_word_pos += "NULL "
            else:
                right_word += "%s " % words[k + 1]
                right_word_pos += "%s " % pos[k + 1]

        left.append(left_word)
        left_pos.append(left_word_pos)
        right.append(right_word)
        right_pos.append(right_word_pos)

    data["left_word"] = left
    data["left_pos"] = left_pos
    data["right_word"] = right
    data["right_pos"] = right_pos
    # data_util.save_data(data, "result_train_test/test_data_end.csv")
    # data_util.save_data(data, "data/test_data_end.csv")
    # data_util.save_data(data, "data/train_data_end.csv")
    data_util.save_data(data, outputFile)


###加入标签
def set_sent_label(inputFile,outputFile):
    #train_data_file_path="result_train_test/train14120_all.csv"
    # train_data_file_path = "result_train_test/test_data_end.csv"
    # train_data_file_path = "data/train_data_end.csv"
    # train_data_file_path = "data/test_data_end.csv"
    data_util = DataUtil()

    data = data_util.load_data1(inputFile)
    label = []
    #     print len(data["sentiment_word"])
    #     print type(data.loc[1,"sentiment_word"].encode("utf8"))

    for i in range(len(data)):
        # 第i行
        sents = data.loc[i, "sentiment_word"]
        themes = data.loc[i, "theme"]
        words = data.loc[i, "words"]

        sent = sents.split(";")[:-1]
        theme = themes.split(";")[:-1]
        la = ""
        for item in words.split(" "):
            if item in sent:
                la += "%s " % ("S")
            elif item in theme:
                la += "%s " % ("T")
            elif item==",":
                la+="%s " % (",")
            else:
                la += "%s " % ("O")

        label.append(la)
    # for i in list:
    #         print i
    print len(label)
    data["simple_labels"] = label

    # output_file_path = 'result_train_test/test_data_end02%d.csv' % len(data)
    # output_file_path = 'data/train_data_label.csv'
    # output_file_path = 'data/test_data_label.csv'
    data_util.save_data(data, outputFile)


# def get_word2vec():
#     data_util=DataUtil()
#     data = data_util.load_data1("train_data_1_17651.csv")
#     word2_vec=[]
# 
#     for i in range(len(data)):
#         words = data.loc[i,"words"].split(" ")
#         array1 = []
# 
#         for item in words:
#             if  item  in model.vocab:
#                 array1.append(model[item])
#             else:
#                 array1.append([0])
#         word2_vec.append(array1)
#     data["word2vec"] = word2_vec
#     data.save('word2vec_feature.csv')


###将训练数据存储为模型输入格式
def csv_to_data(inputFile,outputFile):
    # train_data_file_path = "data/train_data_label.csv"
    # train_data_file_path = "data/test_data_end.csv"

    data_util = DataUtil()
    data = data_util.load_data1(inputFile)
    sentence = []
    for i in range(len(data)):
        word = data.loc[i, "words"].split(" ")
        pos = data.loc[i, "pos"].split(" ")[:-1]
        pars = data.loc[i, "parser"].split(" ")[:-1]
        label = data.loc[i, "simple_labels"].split(" ")[:-1]
        #         nearest_distance_to_sent=data.loc[i,"nearest_distance_to_sent"].split(" ")[:-1]
        #         nearest_distance_to_theme=data.loc[i,"nearest_distance_to_theme"].split(" ")[:-1]
        parents_word = data.loc[i, "parents_word"].split(" ")[:-1]
        parents_pos = data.loc[i, "parents_pos"].split(" ")[:-1]
        parents_parser = data.loc[i, "parents_parser"].split(" ")[:-1]
        # left_word = data.loc[i, "left_word"].split(" ")[:-1]
        # left_pos = data.loc[i, "left_pos"].split(" ")[:-1]
        # right_word = data.loc[i, "right_word"].split(" ")[:-1]
        # right_pos = data.loc[i, "right_pos"].split(" ")[:-1]
        #         relyT=data.loc[i,"relyT"].split(" ")[:-1]
        #         relyS=data.loc[i,"relyS"].split(" ")[:-1]
        sent = []
        for j in range(len(word)):
            train_word = word[j] + " "
            train_word += "%s %s %s %s %s %s " % (pos[j],
                                                              pars[j],
                                                              parents_word[j],
                                                              parents_pos[j],
                                                              parents_parser[j],
                                                              # left_word[j],
                                                              # left_pos[j],
                                                              # right_word[j],
                                                              # right_pos[j],
                                                              label[j]
                                                              )


            train_word.strip(" ")
            sent.append(train_word)
        sentence.append(sent)

    # train_data = "data/traindata_model_train.data"
    # train_data = "data/testdata_model_test.data"

    f = open(outputFile, "a")

    for item in sentence:
        for it in item:
            f.write(it + "\n")
        f.write("\n")

    f.close()

###将测试数据存储为模型输入格式
def csv_to_data2(inputFile,outputFile):

    # train_data_file_path = "data/test_data_end.csv"
    data_util = DataUtil()
    data = data_util.load_data1(inputFile)
    sentence = []
    for i in range(len(data)):
        id = data.loc[i, "row_id"]
        word = data.loc[i, "new_words"].split(" ")
        pos = data.loc[i, "pos"].split(" ")[:-1]
        pars = data.loc[i, "parser"].split(" ")[:-1]
        # label = data.loc[i, "simple_labels"].split(" ")[:-1]
        #         nearest_distance_to_sent=data.loc[i,"nearest_distance_to_sent"].split(" ")[:-1]
        #         nearest_distance_to_theme=data.loc[i,"nearest_distance_to_theme"].split(" ")[:-1]
        parents_word = data.loc[i, "parents_word"].split(" ")[:-1]
        parents_pos = data.loc[i, "parents_pos"].split(" ")[:-1]
        parents_parser = data.loc[i, "parents_parser"].split(" ")[:-1]
        # left_word = data.loc[i, "left_word"].split(" ")[:-1]
        # left_pos = data.loc[i, "left_pos"].split(" ")[:-1]
        # right_word = data.loc[i, "right_word"].split(" ")[:-1]
        # right_pos = data.loc[i, "right_pos"].split(" ")[:-1]
        #         relyT=data.loc[i,"relyT"].split(" ")[:-1]
        #         relyS=data.loc[i,"relyS"].split(" ")[:-1]
        sent = []
        print id
        for j in range(len(word)):
            train_word = word[j] + " "

            train_word += "%s %s %s %s %s " % (pos[j],
                                                              pars[j],
                                                              parents_word[j],
                                                              parents_pos[j],
                                                              parents_parser[j],
                                                              # left_word[j],
                                                              # left_pos[j],
                                                              # right_word[j],
                                                              # right_pos[j],
                                                              )

            train_word.strip(" ")
            sent.append(train_word)
        sentence.append(sent)

    # train_data = "data/traindata_model_train.data"
    # train_data = "data/testdata_model_test.data"

    f = open(outputFile, "a")

    for item in sentence:
        for it in item:
            f.write(it + "\n")
        f.write("\n")

    f.close()


# 将训练数据随机切割成训练集和测试集
def split_data(inputFile,outputFile1,outputFile2):
    # 将数据随机切割成训练集和测试集
    # train_data_file_path = "data/train_data_1_17651.csv"

    data_util = DataUtil()
    data = data_util.load_data1(inputFile)

    train_data, test_data = data_util.split_train_test(data, train_split=0.8)

    print train_data.shape
    print test_data.shape
    # data_util.print_data_detail(test_data)
    # 保存数据
    # data_util.save_data(train_data, 'result_train_test/train_data_%d.csv' % len(train_data))
    # data_util.save_data(test_data, 'result_train_test/test_data_%d.csv' % len(test_data))
    data_util.save_data(train_data, outputFile1)
    data_util.save_data(test_data, outputFile2)

##测试数据分词
def fenci_test(inputFile,outputFile):
    data_util = DataUtil()

    data = data_util.load_data1(inputFile)
    print  ('分词')
    # data['words'] = data['contents'].apply(data_util.segment_sentence)
    data['words'] = data['content'].apply(data_util.segment_sentence)
    print data["words"]
    print ('保存数据')
    words_new=[]
    for i in range(len(data)):
        words = data.loc[i,"words"]
        if len(words)!=0:
            if words[0]==",":
                words=words[1:]
            elif words[-1]==",":
                words=words[:-1]
        words_new.append(words.strip(",").strip(" "))
    data["new_words"]=words_new
    # 保存数据
    # data_util.save_data(data, 'data/train_data_0_%d.csv' % len(data))

    # 将其他字段有空数值的数据去除
#     data = data_util.processing_na_value(data, clear_na=True)
    # output_file_path = 'data/test_fc_data.csv'
    print '去除其他字段有空数值的数据之后，剩下%d条句子,输出到：%s' % (len(data), outputFile)
    # 保存数据
    
    data_util.save_data(data, outputFile)

# def process_test_data():
#     train_data_file_path = "data/test_data.csv"
#     data = pd.read_csv(train_data_file_path,
#                        sep="\t",
#                        encoding="utf8",
#                        names=["row_id"])
#
#     data_util = DataUtil()







###计算与主题、情感距离
def location_to_sent_and_theme():
    train_data_file_path = "result_now/label_data_simple_17651.csv"
    data_util = DataUtil()
    data = data_util.load_data01(train_data_file_path)

    nearest_distance_sent = []
    nearest_distance_theme = []
    for i in range(len(data)):
        labels = data.loc[i, "simple_labels"].split(" ")[:-1]  # 获取label的列表
        #         print labels
        sents_index = []
        themes_index = []
        others_index = []
        distance_sent = []
        distance_theme = []
        for j in range(len(labels)):
            if labels[j] == "S":
                sents_index.append(j)
            elif labels[j] == "T":
                themes_index.append(j)
            else:
                others_index.append(j)

        # 计算当前词与sent的距离
        for j in range(len(labels)):
            if len(sents_index) > 0:
                if labels[j] == "S":
                    distance_sent.append(0)
                else:
                    len0 = []
                    for k in range(len(sents_index)):
                        len0.append(sents_index[k] - j)
                    distance_sent.append(len0)
            else:
                distance_sent.append("NULL")
        # 计算当前词与theme的距离
        for j in range(len(labels)):
            if len(themes_index) > 0:
                if labels[j] == "T":
                    distance_theme.append(0)
                else:
                    len1 = []
                    for k in range(len(themes_index)):
                        len1.append(themes_index[k] - j)
                    distance_theme.append(len1)
            else:
                distance_theme.append("NULL")
        print ("*********************")
        print distance_sent
        #         print distance_theme
        nearest_sent = ""
        nearest_theme = ""
        # 计算最近的sent距离
        for item in distance_sent:
            if item == "NULL":
                nearest_sent += ("NULL ")
            else:
                if item == 0:
                    nearest_sent += "0 "
                else:
                    absList = map(abs, item)
                    l1 = sorted(absList)
                    nearest_sent += "%d " % (l1[0])

        # 计算最近的theme距离
        for item in distance_theme:
            if item != "NULL":
                if item == 0:
                    nearest_theme += "0 "
                else:
                    #                     print i
                    #                     print item
                    absList = map(abs, item)
                    l = sorted(absList)
                    #                 print len(l)
                    nearest_theme += "%d " % (l[0])
            elif item == "NULL":
                nearest_theme += ("NULL ")
        nearest_distance_sent.append(nearest_sent)
        nearest_distance_theme.append(nearest_theme)
        print nearest_sent
        print ("*********************")
    # print nearest_distance_theme
    print("******")
    #     print len(nearest_distance_sent)
    data["nearest_distance_to_sent"] = nearest_distance_sent
    data["nearest_distance_to_theme"] = nearest_distance_theme
    data_util.save_data(data, "result_now/nearest_distance_to_sent_theme.csv")





if __name__ =='__main__':

##训练数据
    # split_data("data/train_data_1.csv",'result_train_test/train_data.csv','result_train_test/test_data.csv')
    # preprocess_train_data("data/train_data.xlsx","data/train_data_2.csv")
    # pos_and_par("data/train_data_2.csv","data/train_pos_pars.csv")
    # parents_feature("data/train_pos_pars.csv","data/train_parents_feature.csv")
    # get_word_left_right("data/train_parents_feature.csv","data/train_data_end.csv")
    # set_sent_label("data/train_parents_feature.csv","data/train_data_label2.csv")
    # # get_word2vec()
    # csv_to_data("data/train_data_label2.csv","data/traindata_model.data")
# 
# ###测试数据
    fenci_test("data/test_df_data_0_20000.csv", "data/test_fc_data.csv")
#     pos_and_par("data/test_fc_data.csv", "data/test_data_pos_pars.csv")
#     parents_feature("data/test_data_pos_pars.csv", "data/test_data_parents_feature.csv")
    # get_word_left_right("data/test_data_parents_feature.csv", "data/test_data_end.csv")
    csv_to_data2("data/test_data_parents_feature.csv", "data/testdata_model.data")


