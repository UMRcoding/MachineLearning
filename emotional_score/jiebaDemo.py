import csv
import jieba
from pandas._libs.internals import defaultdict

'''
分词 => 预处理(去掉停用词，计算得分) => 情感分析（正向 负向）
根据情感词分组  找到情感词  副词 否定词
情感词字典 Boson
程度副词字典 degree 不同副词的乘法系数不同
否定词字典 negword ,否定词 *（-1）
停用词字典 stopword
'''

def str_to_dict(word_list):
    '''
    将分词后的str转为字典，key为词，value为该词在列表中的索引，索引相当于词语在文档中出现的位置
    :param word_list:
    :return:
    '''
    # 将 str 以 / 分割 转为列表
    word_list = word_list.split('/')
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data


def readwords_list(filepath):
    '''
    读取文件函数，以行的形式读取词表，返回列表
    :param filepath: 路径地址
    :return:
    '''
    wordslist = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]
    return wordslist

def readwords_dict(filepath):
    '''
    读取文件函数，返回字典
    :param filepath: 路径地址
    :return:
    '''
    wordslist = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]
    result_dict = defaultdict()
    # 读取字典文件每一行内容，将其转换为字典对象，key为情感词，value为对应的分值
    for s in wordslist:
        if s != '':
            # 每一行内容根据空格分割，索引0是词，索引1是分值
            if '\t' in s:
                result_dict[s.split('\t')[0]] = s.split('\t')[1]  # 以\t分割
            else:
                result_dict[s.split(' ')[0]] = s.split(' ')[1]  # 以空格分割
    return result_dict

def cutsentences(sentences):
    '''
    分词的实现
    :param sentences:
    :return:
    '''
    # print('\n' + '原句子为：' + sentences)
    cut_result_list = jieba.lcut(sentences.strip())  # 精确模式
    # print('分词后：' + "/ ".join(cut_result_list))
    return cut_result_list

def stopword_sentence(cut_result_list, stop_words):
    '''
    删去停用词
    :param cut_result_sentence: 分词后的结果
    :param stop_words:
    :return:
    '''
    stop_result_sentence = ''
    for word in cut_result_list:  # for循环遍历分词后的每个词语
        if word not in stop_words and word != '':  # 判断分词后的词语是否在停用词表内
            stop_result_sentence += word
            stop_result_sentence += "/"
    # print('删去停用词后：' + stop_result_sentence + '\n')
    return stop_result_sentence

def word_classify(sentence_dict,sentiment_dist,neg_words,degree_dic):
    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    sen_word = dict()
    not_word = dict()
    degree_word = dict()

    for word in sentence_dict.keys():
        if word in sentiment_dist.keys() and word not in neg_words and word not in degree_dic.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[sentence_dict[word]] = sentiment_dist[word]
        elif word in neg_words and word not in degree_dic.keys():
            # 分词结果中在否定词列表中的词
            not_word[sentence_dict[word]] = -1
        elif word in degree_dic.keys():
            # 分词结果中在程度副词中的词
            degree_word[sentence_dict[word]] = degree_dic[word]
    return sen_word, not_word, degree_word

def socre_sentiment(sentences,sentiment_words, neg_words, degree_words):
    '''
    设置初始权重 W为1，
    用权重W * 该情感词的情感值作为得分（用score记录），然后判断与下一个情感词之间是否有程度副词及否定词
    如果有否定词将W*-1，如果有程度副词，W*程度副词的程度值
    此时的W作为遍历下一个情感词的权重值，循环直到遍历完所有的情感词，每次遍历过程中的得分score加起来的总和就是这篇文档的情感得分。
    :param sentences: 待分类句子
    :param sentiment_words: 情感词
    :param neg_words: 否定词
    :param degree_words: 程度副词
    :return:
    '''
    W = 1  # 权重初始化为1
    score = 0
    sentiment_index = -1    # 情感词下标初始化
    # 情感词的位置下标集合
    sentiment_index_list = list(sentiment_words.keys())
    # 遍历分词结果(遍历分词结果是为了定位两个情感词之间的程度副词和否定词)
    for i in range(0, len(sentences)):
        # 如果是情感词（根据下标是否在情感词分类结果中判断）
        if i in sentiment_words.keys():
            # 权重*情感词得分
            score += W * float(sentiment_words[i])
            # 情感词下标加1，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list) - 1:
                # 判断当前的情感词与下一个情感词之间是否有程度副词或否定词
                for j in range(sentiment_index_list[sentiment_index], sentiment_index_list[sentiment_index + 1]):
                    # 更新权重，如果有否定词，取反
                    if j in neg_words.keys():
                        W *= -1
                    elif j in degree_words.keys():
                        # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
                        W *= float(degree_words[j])
        # 定位到下一个情感词
        if sentiment_index < len(sentiment_index_list) - 1:
            i = sentiment_index_list[sentiment_index + 1]
    return score

if __name__ == '__main__':
    data_list = []  # 存放处理结果
    # 读取影评文件
    sentence_filepath = 'data/影评.txt'
    sentences_list = readwords_list(sentence_filepath)

    # 读取停用词文件
    stopwords_filepath = 'data/stopwords_1208.txt'
    stop_list = readwords_list(stopwords_filepath)

    # 读取情感字典文件
    sentiment_filepath = 'data/BosonNLP_sentiment_score.txt'
    sentiment_dict = readwords_dict(sentiment_filepath)

    # 读取否定词文件
    neg_filepath = 'data/negwords_70.txt'
    neg_list = readwords_list(neg_filepath)

    # 读取程度副词文件
    degree_filepath = 'data/degreeDict.txt'
    degree_dict = readwords_dict(degree_filepath)


    for sentence in sentences_list:
        data = {}
        data['原句'] = sentence

        # 1. 分词
        cut_results_list = cutsentences(sentence)
        data['分词结果'] = "/".join(cut_results_list)

        # 2. 去掉停用词，过滤掉某些字或词
        stoped_results = stopword_sentence(cut_results_list, stop_list)
        stoped_results_dict = str_to_dict(stoped_results)
        data['去掉停用词结果'] = stoped_results

        # 3. 对分词结果分类
        sen_word, not_word, degree_word = word_classify(stoped_results_dict, sentiment_dict, neg_list, degree_dict)

        # 4. 找出情感词、否定词和程度副词
        score = socre_sentiment(stoped_results_dict,sen_word,not_word,degree_word)
        data['情感得分'] = score

        data_list.append(data)

    # 保存CSV
    # 创建文件对象
    file_path = 'Result.csv'
    f = open (file_path,'w', encoding='utf-8-sig', newline='') #newline=''防止空行
    # 基于对象构建CSV
    csv_write = csv.writer(f)
    # 构建列表头
    csv_write.writerow(['原句','分词结果','去掉停用词结果','情感得分'])
    # 写入CSV文件
    for data in data_list:
        csv_write.writerow([data["原句"],data["分词结果"],data["去掉停用词结果"],data["情感得分"]])
    print("结果写入成功" + file_path)