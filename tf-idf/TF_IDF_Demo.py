import csv
import math
import operator
from collections import defaultdict
import jieba

class ToDo():
    def __init__(self):
        self.comment_path = 'data/comment.csv' # 文件保存位置

    def file_read(self):
        data_list = []
        try:
            # 字典
            with open(self.comment_path, 'r', encoding='utf-8-sig') as fp:
                reader = csv.DictReader(fp)
                for item in reader:
                    data_list.append(item['评论内容'])
            return data_list
        except Exception as e:
            print("\033[1;31m文件读取出错:\033[0m" + self.comment_path)
            print(e)

    def readwords_list(self, filepath):
        '''
        读取文件函数，以行的形式读取词表，返回列表
        :param filepath: 路径地址
        :return:
        '''
        wordslist = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return wordslist

    def readwords_dict(self, filepath):
        '''
        读取文件函数，返回字典
        :param filepath: 路径地址
        :return:
        '''
        wordslist = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
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

    def cutsentences(self, sentences):
        '''
        分词的实现
        :param sentences:
        :return:
        '''
        # print('\n' + '原句子为：' + sentences)
        cut_result_list = jieba.lcut(sentences.strip())  # 精确模式
        # print('分词后：' + "/ ".join(cut_result_list))
        return cut_result_list

    def stopword_sentence(self, cut_result_list, stop_words):
        '''
        删去停用词
        :param cut_result_sentence: 分词后的结果
        :param stop_words:
        :return:
        '''
        stop_result_sentence = []
        for word in cut_result_list:  # for循环遍历分词后的每个词语
            if word not in stop_words and word != ' ' and word != '\n':  # 判断分词后的词语是否在停用词表内
                stop_result_sentence.append(word)
                # stop_result_sentence += "/"
        # print('删去停用词后：' + stop_result_sentence + '\n')
        return stop_result_sentence

    # 特征选择：TF-IDF算法
    def feature_select(self, list_words):
        # 总词频统计，即计算一共有多少词
        doc_frequency = defaultdict(int)
        for list_word in list_words:
            for i in list_word:
                doc_frequency[i] += 1

        #  计算每个词的词频即TF值：出现次数/总次数
        word_tf = {}
        for i in doc_frequency:
            word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

        #   计算逆文档频率 IDF：Log(语料库的文档总数/（包含该词的文档数+1）)，
        #   IDF大概为丑化因子，用来区别在多少文档出现，权重小即区分度大。在word_idf数值中已经做丑化，所以直接相乘
        word_idf = {}
        doc_num = len(list_words)  # 总文章数
        word_doc = defaultdict(int)  # 包含该词的文章数
        for i in doc_frequency:
            for j in list_words:
                if i in j:
                    word_doc[i] += 1
        for i in doc_frequency:
            word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

        # 计算每个词的 tf * idf
        word_tf_idf = {}
        for i in doc_frequency:
            word_tf_idf[i] = word_tf[i] * word_idf[i]

        #   提取多少特征，需要先对字典由大到小排序
        dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
        return dict_feature_select

    def TF_DEF(self):
        # 读取停用词文件
        stopwords_filepath = 'data/stopwords_1208.txt'
        stop_list = self.readwords_list(stopwords_filepath)

        sentence_cuted = []
        sentences_list = self.file_read()

        for sentence in sentences_list:
            # 1. 分词
            cut_results_list = self.cutsentences(sentence)
            # 2. 去掉停用词，过滤掉某些字或词
            stoped_results = self.stopword_sentence(cut_results_list, stop_list)
            sentence_cuted.append(stoped_results)
        features = self.feature_select(sentence_cuted)
        print(features[:20])

if __name__ == '__main__':
    ToDo().TF_DEF()
