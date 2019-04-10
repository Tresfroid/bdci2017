# -*- coding: utf-8 -*-  #

import sys
import csv
# reload(sys)
# sys.setdefaultencoding('utf-8')
import numpy as np
import pandas as pd
import logging
# from pyltp import Segmentor,Postagger,Parser
# from jieba_util import Jieba_Util
import  re
import jieba
jieba.load_userdict("dict_add/word4seg.txt")
# import gensim
# model = gensim.models.Word2Vec.load("D:/word2vec/Word60.model")
# from cryptography.hazmat.primitives.serialization import Encoding


# segmentor = Segmentor()
# segmentor.load('LTP\ltp_data\cws.model')
# postagger = Postagger()
# postagger.load("LTP\ltp_data\pos.model")  #词性标注
# parser = Parser()
# parser.load("LTP\ltp_data\parser.model")

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
    
    # def print_data_detail(self, data, has_sentiment_anls=True):
    #     '''
    #         展示数据的详细信息
    #     :param data: Dateframe对象
    #     :param has_sentiment_anls: 是否有sentiment_anls字段
    #     :return: 无
    #     '''
    #
    #     logging.debug('data的个数为：%d' % (len(data)))
    #     logging.debug('data的sample数据：')
    #     logging.debug(data.head())
    #
    #     logging.debug('data的target和个数分别为：')
    #     logging.debug(data['sentiment_anls'].value_counts())
    #     """if has_sentiment_anls:
    #         logging.debug('统计每个Target下各个类型立场的数量...')
    #         group = data.groupby(by=['TARGET', 'STANCE'])
    #         logging.debug(group.count())
    #     else:
    #         logging.debug('没有STANCE字段')"""
    #
    #     logging.debug('数据各个字段情况...')
    #     for column in data.columns:
    #         # 统计每个字段是否有数据是空串
    #         # 先将所有空字符串用nan替换
    #         data[column] = data[column].replace(r'^\s*$', np.nan, regex=True)
    #         count_null = sum(data[column].isnull())
    #         if count_null != 0:
    #             logging.warn(u'%s字段有空值，个数：%d,建议使用processing_na_value()方法进一步处理！' % (column, count_null))
    #             null_data_path = 'null_data.csv'
    #             logging.warn(u'将缺失值数据输出到文件：%s' % (null_data_path))
    #             data[data[column].isnull()].to_csv(null_data_path,
    #                                                index=None,
    #                                                encoding='utf8',
    #                                                sep='\t')
    #
    # def processing_na_value(self,data,clear_na=True,fill_na=False,fill_char="NONE_NULL",columns=None):
    #     '''
    #         处理数据的空值
    #
    #     :param data:  Dateframe对象
    #     :param clear_na: bool,是否去掉空值数据
    #     :param fill_na: bool，是否填充空值
    #     :param fill_char: str，填充空置的字符
    #     :param column: list，需要处理的字段，默认为None时，对所有字段处理
    #     :return: Dateframe对象
    #     '''
    #     logging.debug('[def processing_na_value()] 对缺失值进行处理....')
    #     for column in data.columns:
    #         if columns == None or column in columns:
    #             data[column] = data[column].replace(r"^\s*$",np.nan,regex=True)
    #             count_null =sum(data[column].isnull())
    #             if count_null!=0:
    #                 logging.warn(u'%s字段有空值，个数：%d' % (column, count_null))
    #                 if clear_na:
    #                     logging.warn(u'对数据的%s字段空值进行摘除'%(column))
    #                     data = data[data[column].notnull()].copy()
    #                 else:
    #                     if fill_na:
    #                         logging.warn(u'对数据的%s字段空值进行填充，填充字符为：%s'%(column,fill_char))
    #                         data[column] = data[column].fillna(value=fill_char)
    #
    #     return data
    #
    def segment_sentence(self,sentence):
        sentence = re.sub(u"好+", "好", sentence)
        sentence = re.sub(u"棒+", "棒", sentence)
        sentence = re.sub(u"\?+", ",", sentence)
        sentence = re.sub(u"\!+", ",", sentence)
        sentence = re.sub(u"\.+", ",", sentence)
        sentence = re.sub(u"，+", ",", sentence)
        sentence = re.sub(u"？+", ",", sentence)
        sentence = re.sub(u"！+", ",", sentence)
        sentence = re.sub(u"。+", ",", sentence)
        sentence = re.sub(u"\s+", ",", sentence)
        sentence = re.sub(u",+", ",", sentence)

        sentence1 = sentence.split(",")#
        sen=" "
        for item in sentence1:
            clause=jieba.cut(item.strip())
            temp=""
            for i in clause:
                if i in ["很","太","再","有点"]:
                    i=""
                else:
                    temp+="%s "%(i)
            sen +=temp.strip(" ")
            sen=sen+" , "
        sent=sen.strip(",").strip(" ").strip(",").strip(" ")
        return sent
    #
    # def split_train_test(self,data, train_split=0.8):
    #     '''
    #         将数据切分成训练集和验证集
    #
    #     :param data:
    #     :param train_split: float，取值范围[0,1],设置训练集的比例
    #     :return: dev_data,test_data
    #     '''
    #     logging.debug('对数据随机切分成train和test数据集，比例为：%f' % (train_split))
    #     num_train = len(data)
    #     num_dev = int(num_train * train_split)
    #     num_test = num_train - num_dev
    #     logging.debug('全部数据、训练数据和测试数据的个数分别为：%d,%d,%d' % (num_train, num_dev, num_test))
    #     rand_list = np.random.RandomState(0).permutation(num_train)
    #     # print rand_list
    #     # print rand_list[:num_dev]
    #     # print rand_list[num_dev:]
    #     dev_data = data.iloc[rand_list[:num_dev]].sort_index()
    #     test_data = data.iloc[rand_list[num_dev:]].sort_index()
    #     # print dev_data
    #     # print test_data
    #     return dev_data, test_data
    #
    #
    #     # 将数据随机切割成训练集和测试集


def fenci_test(inputFile,outputFile):
    data_util = DataUtil()

    data = data_util.load_data1(inputFile)
    print  ('分词')
    data['words'] = data['contents'].apply(data_util.segment_sentence)
    # print data["words"]
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
    # print '去除其他字段有空数值的数据之后，剩下%d条句子,输出到：%s' % (len(data), outputFile)
    # 保存数据
    
    data_util.save_data(data, outputFile)


if __name__ =='__main__':

# ###测试数据
    fenci_test("data/test_data.csv", "data/test_fc_data.csv")



