import codecs
from data_util import DataUtil


def calculate_tf():
    dict_num = 0
    f1 = codecs.open('dict_add/theme_dict_fu.txt','w',encoding='utf-8')
    f2 = codecs.open('dict_add/sentiment_dict_fu.csv', 'w', encoding='utf-8')
    f3 = codecs.open('dict_add/polarity_dict_fu.csv', 'w', encoding='utf-8')
    f = codecs.open('dict_add/user_dict.txt', 'w', encoding='utf-8')

    # outputFile1 = open('dict_add/theme_dict_fu.csv', 'w')
    # outputFile2 = open('dict_add/sentiment_dict_fu.csv', 'w')
    # outputFile3 = open('dict_add/polarity_dict_fu.csv', 'w')
    # outputFile = open('dict_add/user_dict.txt', 'w')
    train_data_file_path = "data/train_data.csv"
    data_util = DataUtil()
    data = data_util.load_data1(train_data_file_path)

    theme_count = {}
    sentiment_word_count = {}
    polarity_dict = {}
    for i in range(len(data)):
        theme_list = []
        sentiment_word_list= []
        polarity_list = []
        theme = str(data.loc[i,'theme']).split(";")
        sentiment_word = str(data.loc[i,'sentiment_word']).split(";")
        polarity = str(data.loc[i,'polarity']).split(";")
        for t in theme:
            theme_list.append(t)
        theme_list = theme_list[:-1]
        for s in sentiment_word:
            sentiment_word_list.append(s)
        sentiment_word_list = sentiment_word_list[:-1]
        for p in polarity:
            polarity_list.append(p)
        polarity_list = polarity_list[:-1]

        for t in theme_list:
            if t in theme_count:
                theme_count[t] += 1
            else:
                theme_count[t] = 1

        for s in sentiment_word_list:
            if s in sentiment_word_count:
                sentiment_word_count[s] += 1
            else:
                sentiment_word_count[s] = 1
        # print (sentiment_word_list)
        # print len(sentiment_word_list)

        for i in range(len(sentiment_word_list)):
            if sentiment_word_list[i] in polarity_dict:
                break
            else:
                polarity_dict[sentiment_word_list[i]] = polarity_list[i]
        # print "\t".join(theme_list)
        # print "\t".join(sentiment_word_list)
        # print "\t".join(polarity_list)
        # print "\n"



    print ("高频主题词：")
    for t in theme_count:
        if theme_count[t] >50:
            print (t, theme_count[t])
            f1.write(t+ "\n")
            f.write(t+ " 9999 n"+ "\n")
            # outputFile1.write(t)
            # outputFile1.write("\n")
            # outputFile.write(t+' '+"9999"+' n')
            # outputFile.write("\n")
            dict_num+=1

    print ("高频情感词：")
    for s in sentiment_word_count:
        if sentiment_word_count[s]>50 :
            # print s, sentiment_word_count[s]
            f2.write(s +"\n")
            f.write(s +" 9999 a"+ "\n")
            # outputFile2.write(s)
            # outputFile2.write("\n")
            # outputFile.write(s+' '+"9999"+' a')
            # outputFile.write("\n")
            dict_num += 1
    print (dict_num)
    print ("极性词典：")
    for p in polarity_dict:
        # print p, polarity_dict[p]
        f3.write(p +"\t"+ polarity_dict[p]+"\n")
        # outputFile3.write(p +"\t"+ str(polarity_dict[p]))
        # outputFile3.write("\n")



if __name__ =='__main__':
    calculate_tf()