# 开源项目说明

**目的：学术研究**

如果您遇到任何安装或使用问题，可以通过QQ或issue的方式告诉我。同时使用本项目写论文或做其它学术研究的朋友，如果想把自己的研究成果展示在下面，也可以通过QQ或issue的方式告诉我。看到的小伙伴记得点点右上角的Star~

![](https://umrcoding.oss-cn-shanghai.aliyuncs.com/Obsidian/202212041543569.png)



# 机器学习

```
情感分析 句子 正向/负向  得分

tfidf的实现

F12 在console输入document.charset 查看编码方式
UTF-8, GBK, GB18030
```



## 1. 前例

### demo01

```python
import json
import requests
import jsonpath
import scrapy

# 找目标地址
base_url = 'https://www.lagou.com/lbs/getAllCitySearchLabels.json'
headers ={
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-CN;q=0.8,en;q=0.7'
}
response = requests.get(base_url, headers=headers, timeout=10)
html_str = response.content.decode()
# json格式转化为python对象
jsonobj = json.loads(html_str)
'''
{
   "state": 1,
   "message": "success",
   "content": {
      "data": {
         "allCitySearchLabels": {
            "A": [{
               "id": 671,
               "name": "安庆",
               "parentId": 541,
               "code": "131800000",
               "isSelected": false
'''
citys = jsonpath.jsonpath(jsonobj,'$..name')
# for city in citys:
#     print(city)
with open('data/citys.txt', 'w', encoding='utf-8-sig') as f:
    content = json.dumps(citys, ensure_ascii=False)
    f.write(content)
```



### Ifanyi

```python
import json
import requests

#       初始化参数
#       获取内容
#       解析
#       运行

class Ifanyi(object):
    # # 将JSON转换为字典
    # dist_data = json.loads(data)
    # print(dist_data)

    # print(*data_href, sep="\n")
    def __init__(self,word):
        self.url = "start_url"
        self.word = word
        self.headers = "headers"
        # 构造内容
        self.from_data = {
            'from': '',
            'to': '',
            "q": self.word
        }

    def get_data(self):
        response = requests.get(self.start_url, headers=self.headers, timeout=10, proxies=self.proxies)
        return response.content

    def parse_data(self,data):
        # 将JSON转换为字典
        dist_data = json.loads(data)
        try:
            print(dist_data['a']['b'])
        except:
            print('err')

    def run(self):
        data = self.get_data()
        self.parse_data(data)

if __name__ == '__main__':
    # 由类生成对象
    Ifanyi = Ifanyi("中国")
    Ifanyi.run()
```



### tkinter

```python
import tkinter

import ttkbootstrap as tk
from ttkbootstrap.constants import *

# 创建窗口对象
root = tk.Window(themename="minty")
root.geometry('500x500')
root.title('scrapy')
root.wm_attributes('-topmost',1)

url_str_var = tkinter.StringVar()
Max_page = tkinter.StringVar()

client_str_var = tk.IntVar()

b1 = tk.Button(root, text="按钮1", bootstyle="success")
b1.pack(side=LEFT, padx=5, pady=10)

b2 = tk.Button(root, text="Submit", bootstyle="info-outline")
b2.pack(side=LEFT, padx=5, pady=10)

root.mainloop()
```



### Thread

```python
import time
from functools import partial
from threading import Thread


def output(content):
    while True:
        print(content,end='',flush=True)
        time.sleep(1)

# 偏函数，基于已有函数，定下部分参数
output1 = partial(output,content='A')
output2 = partial(output,content='B')

def main():
    # t1 = Thread(target=output1)
    # N元组 tuple
    t1 = Thread(target=output1, args=('Pong',))
    t1.start()

    t2 = Thread(target=output2)
    t2.start()

if __name__ == '__main__':
    main()
```



### scrapy

```cmd
scrapy startproject spider
cd spider
scrapy genspider cnvd cnvd.org.com

scrapy crawl cnvd

# cnvdScrapy为项目路径
```

**settings**

```python
# U-A请求头
USER_AGENT = 'User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;'

# 不服从爬虫协议，解决拒绝访问
ROBOTSTXT_OBEY = False

# 对同一网址延迟请求的秒数，受RANDOMIZE_DOWNLOAD_DELAY影响(默认开启)，实际延迟时间为[0.5 * DOWNLOAD DELAY , 1.5 * DOWNLOAD_DELAY]
DOWNLOAD_DELAY = 5

# 每个域名能够被执行的最大并发数 默认8
# CONCURRENT_REQUESTS_PER_DOMAIN = 16

# 能被单个IP处理的并发请求数，默认0 代表无限制。
# I、如果不为零，那CONCURRENT_REQUESTS_PER_DOMAIN将被忽略，即并发数的限制是按照每个IP来计算，而不是每个域名
# II、该设置也影响DOWNLOAD_DELAY，如果该值不为零，那么DOWNLOAD_DELAY下载延迟是限制每个IP而不是每个域
# CONCURRENT_REQUESTS_PER_IP = 16

# cookie（默认启用）
# COOKIES_ENABLED = False

# 开启管道
ITEM_PIPELINES = {
    'sprider.pipelines.SpriderPipeline': 300,
}
```



## 2. 数据清洗

### 2.1 分词

```python
'''
分词 => 预处理(去掉停用词，计算得分) => 情感分析（正向 负向）
根据情感词分组  找到情感词  副词 否定词
情感词字典 Boson
程度副词字典 degree 不同副词的乘法系数不同
否定词字典 negword ,否定词 *（-1）
停用词字典 stopword
'''
```



### jieba分词

```python
# data/dict.txt内容
按时 5
阿三的 2 ne
收到 1 s
乒乓球拍卖完 2 k


import jieba
jieba.load_userdict("data/dict.txt")

test = (
"阿三大苏打撒的 \n"
)

# 全模式  速度快，但不能解决歧义
word_list1 = jieba.cut(test, cut_all=True)
print("Full mode:" + "/".join(word_list1))

# 精确模式 默认 适合文本分析
word_list2 = jieba.cut(test, cut_all=False)
print("Accurate mode:" + "/".join(word_list2))

# 搜索模式 在精确模式基础上，对长词再次划分，提高召回率，适合搜索引擎分词
word_list3 = jieba.cut_for_search(test)
print("Search mode:" + "/".join(word_list3))

# 结果
Full mode:阿/三/大苏打/苏打/撒/的// //
Accurate mode:阿三/大苏打/撒/的/ /
Search mode:阿三/苏打/大苏打/撒/的/ /
```



### 停用词

```python
import jieba

def stopwordslist(filepath):  # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath, 'r').readlines()]  # 以行的形式读取停用词表，同时转换为列表
    return stopword

def cutsentences(sentences):  # 定义函数实现分词
    print('原句子为：' + sentences)
    cutsentence = jieba.lcut(sentences.strip())  # 精确模式
    print('\n' + '分词后：' + "/ ".join(cutsentence))
    stopwords = stopwordslist(filepath)  # 这里加载停用词的路径
    lastsentences = ''
    for word in cutsentence:  # for循环遍历分词后的每个词语
        if word not in stopwords:  # 判断分词后的词语是否在停用词表内
            if word != '\t':
                lastsentences += word
                lastsentences += "/ "
    print('\n' + '去除停用词后：' + lastsentences)

filepath = 'data/stopwords_1208.txt'
stopwordslist(filepath)
# sentences = 'data/影评.txt'
sentences = '太烂了！很多年没看过这么烂的电影了，已经无法形容烂的程度，开场不到二十分钟我就睡着了，睡醒后离场，浪费钱和时间'
cutsentences(sentences)
```



## 2.2 缺失值处理

### **缺失值判断**

```python
# isnull notnull
import numpy as np
import pandas as pd

s = pd.Series([12,33,45,23,np.nan,np.nan,66,54,np.nan,99])
print(s.isnull())
# 结果
0    False
1    False
2    False
3    False
4     True
5     True
6    False
7    False
8     True
9    False
dtype: bool

print(s.notnull())
# 结果
0     True
1     True
2     True
3     True
4    False
5    False
6     True
7     True
8    False
9     True
dtype: bool
```

```python
df = pd.DataFrame({'value1':[12,33,45,23,np.nan,np.nan,66,54,np.nan,99],
                   'value2':['a','b','c','d','e',np.nan,np.nan,'f','g',np.nan]})
print(df.isnull())
# 结果
   value1  value2
0   False   False
1   False   False
2   False   False
3   False   False
4    True   False
5    True    True
6   False    True
7   False   False
8    True   False
9   False    True

print(df['value1'].notnull())
# 结果
0     True
1     True
2     True
3     True
4    False
5    False
6     True
7     True
8    False
9     True
Name: value1, dtype: bool
```



**缺失值筛选**

```python
import numpy as np
import pandas as pd
s = pd.Series([12,33,45,23,np.nan,np.nan,66,54,np.nan,99])
s2 = s[s.isnull() == False]
print(s2)
# 结果
0    12.0
1    33.0
2    45.0
3    23.0
6    66.0
7    54.0
9    99.0
dtype: float64

# 删除缺失值
s.dropna(inplace=True)
print(s)
# 结果
0    12.0
1    33.0
2    45.0
3    23.0
6    66.0
7    54.0
9    99.0
dtype: float64
```

```python
import numpy as np
import pandas as pd
df = pd.DataFrame({'value1':[12,33,45,23,np.nan,np.nan,66,54,np.nan,99],
                   'value2':['a','b','c','d','e',np.nan,np.nan,'f','g',np.nan]})
df2 = df[df['value2'].notnull()]
print(df2)
# 结果
   value1 value2
0    12.0      a
1    33.0      b
2    45.0      c
3    23.0      d
4     NaN      e
7    54.0      f
8     NaN      g

# 删除缺失值
df2 = df['value1'].dropna()
print(df2)
print(df2)
# 结果
0    12.0
1    33.0
2    45.0
3    23.0
6    66.0
7    54.0
9    99.0
Name: value1, dtype: float64
```



**缺失值的填充**

```python
# 均值mean 中位数median 众数mode
import numpy as np
import pandas as pd

s = pd.Series([12,33,45,23,np.nan,np.nan,66,54,np.nan,99])
av = s.mean()
print(s.fillna(av, inplace=False))
```



## 2.3 离群点筛选

```py
# 定义范围  1）数据可视化法 高维失效  2）基于统计的异常点检测  阈值=均值+-2 *标准差   3）基于距离 马氏距离
import numpy as np
import pandas as pd

data = pd.Series(np.random.rand(10000)*100)
mean = data.mean()
std = data.std()
print(mean)
print(std) #标准差

error = data[np.abs(data-mean)>3*std]
data2 = data[np.abs(data-mean)<=3*std]
print(data2)
# 结果
0       74.510734
1       88.825108
2       62.303807
3       73.056244
4       69.918117
          ...    
9995    94.298266
9996     0.970466
9997    84.015327
9998    77.718017
9999    99.467432
Length: 10000, dtype: float64
```



## 2.4 情感分析

找出文档中的情感词、否定词以及程度副词，然后判断每个情感词之前是否有否定词及程度副词，将它之前的否定词和程度副词划分为一个组，如果有否定词将情感词的情感权值乘以-1，如果有程度副词就乘以程度副词的程度值，最后所有组的得分加起来，大于0的归于正向，小于0的归于负向。

BosonNLP是基于微博、新闻、论坛等数据来源构建的情感词典，因此拿来对其他类别的文本进行分析效果可能不好





# 3. 特征工程

## 3.1 特征选择

### 最大最小规范化

```python
# 数据变换  数据标准化  小数缩放 -> 12,34,56,1,100放入[-1，1]  -> 以最大值为依据进行缩放
# 最小最大标准化  -> (v(i) - min(v(i)))/max(v(i)-min(v(i))) -> [0,1]
# 标准差标准化  v(i) = (v(i) - mean(v))/std(v)
import numpy as np
from sklearn import preprocessing

X = np.array([[1,-1,2,3],
             [2,0,0,1],
             [0,1,-1,2]
             ])

# 按列进行归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(X)
print(x_minmax)
# 结果
[[0.5        0.         1.         1.        ]
 [1.         0.5        0.33333333 0.        ]
 [0.         1.         0.         0.5       ]]


X_scaled = preprocessing.scale(X)
print(X_scaled)
# 结果
[[ 0.         -1.22474487  1.33630621  1.22474487]
 [ 1.22474487  0.         -0.26726124 -1.22474487]
 [-1.22474487  1.22474487 -1.06904497  0.        ]]

# 按列求平均值
print(X_scaled.mean(axis=0))
# 结果
[0. 0. 0. 0.]
```



### TF-IDF算法

单纯以`TF-IDF`算法衡量一个词的重要性不够全面，无法体现词的位置信息

```python
import math
import operator
from collections import defaultdict

def loadDataSet():
    # 切分的词条
    dataset = [
        ['太烂','了','很多年','没', '看过','这么','烂','的','电影','了'],
        ['已经','无法形容','烂','的','程度'],
        ['开场','不到','二十分钟','我','就','睡着','了'],
        ['睡醒','后','离场','浪费','钱','和','时间']
    ]
    # 类型标签向量  1：好   0：不好
    classVec = [0,1,0,1,0,1]
    return dataset,classVec

# 特征选择：TF-IDF算法
def feature_select(list_words):
    # 总词频统计，即计算一共有多少词
    doc_frequency = defaultdict(int)
    for list_word in list_words:
        for i in list_word:
            doc_frequency[i] += 1

#   计算每个词的词频即TF值：出现次数/总次数
    word_tf = {}
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

#   计算逆文档频率 IDF：Log(语料库的文档总数/（包含该词的文档数+1）)，
#   IDF大概为丑化因子，用来区别在多少文档出现，权重小即区分度大。在word_idf数值中已经做丑化，所以直接相乘
    word_idf = {}
    doc_num = len(list_words)       # 总文章数
    word_doc = defaultdict(int)     # 包含该词的文章数
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
    dict_feature_select = sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select

if __name__ == '__main__':
    dataset,label_list = loadDataSet()
    features = feature_select(dataset)
    print(features)
    print(len(features))
    
# 结果
[('了', 0.029760214391563535), ('太烂', 0.02390162691586018), ('很多年', 0.02390162691586018), ('没', 0.02390162691586018), ('看过', 0.02390162691586018), ('这么', 0.02390162691586018), ('电影', 0.02390162691586018), ('已经', 0.02390162691586018), ('无法形容', 0.02390162691586018), ('程度', 0.02390162691586018), ('开场', 0.02390162691586018), ('不到', 0.02390162691586018), ('二十分钟', 0.02390162691586018), ('我', 0.02390162691586018), ('就', 0.02390162691586018), ('睡着', 0.02390162691586018), ('睡醒', 0.02390162691586018), ('后', 0.02390162691586018), ('离场', 0.02390162691586018), ('浪费', 0.02390162691586018), ('钱', 0.02390162691586018), ('和', 0.02390162691586018), ('时间', 0.02390162691586018), ('烂', 0.019840142927709022), ('的', 0.019840142927709022)]
25
```



### sklearn的特征提取

```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba

def cut_word(text):
    text = "".join(list(jieba.cut(text)))
    return text

def chinese_text_count_demo():
    # 获取数据集
    data = [
        'Matt D-Avella  特点：制作精良（博主是film maker） 内容真实  有思想、内涵 推荐视频： 《A Day in the Life of Minimalist》《12 Habits for Life》',
        'Nathaniel Drew 特点：高质量 有深度 内容广泛 推荐视频：《How I Rediscovered Books（In the Digital Age）》《How Learning Languages Changed My View Of the World》',
        'Dan Mace (film maker) 特点：讲故事能力 剪辑能力 创意 推荐视频：《How to edit like Dan Mace》 《The Not Normal Show》',
        '.Johnny Harris  （《Vox-Borders》制作人） 特点：用简单却能吸引你看下去的方式讲述道理或现象 推荐视频：《How to Force Yourself to Learn Stuff》 《How to Remember Your Life》'
        ]

    # 文章分割
    test_list = []
    for i in data:
        test_list.append(cut_word(i))
    print('分词\n',test_list)

    # 特征提取
    # 实例化转换器
    transfer = CountVectorizer(stop_words=['ok'])
    new_data = transfer.fit_transform(test_list)

    names = transfer.get_feature_names_out()

    print('特征名称\n', names)
    print(new_data.toarray())

if __name__ == '__main__':
    chinese_text_count_demo()
    
    
# 结果
分词
 ['Matt D-Avella  特点：制作精良（博主是film maker） 内容真实  有思想、内涵 推荐视频： 《A Day in the Life of Minimalist》《12 Habits for Life》', 'Nathaniel Drew 特点：高质量 有深度 内容广泛 推荐视频：《How I Rediscovered Books（In the Digital Age）》《How Learning Languages Changed My View Of the World》', 'Dan Mace (film maker) 特点：讲故事能力 剪辑能力 创意 推荐视频：《How to edit like Dan Mace》 《The Not Normal Show》', '.Johnny Harris  （《Vox-Borders》制作人） 特点：用简单却能吸引你看下去的方式讲述道理或现象 推荐视频：《How to Force Yourself to Learn Stuff》 《How to Remember Your Life》']
特征名称
 ['12' 'age' 'avella' 'books' 'borders' 'changed' 'dan' 'day' 'digital'
 'drew' 'edit' 'film' 'for' 'force' 'habits' 'harris' 'how' 'in' 'johnny'
 'languages' 'learn' 'learning' 'life' 'like' 'mace' 'maker' 'matt'
 'minimalist' 'my' 'nathaniel' 'normal' 'not' 'of' 'rediscovered'
 'remember' 'show' 'stuff' 'the' 'to' 'view' 'vox' 'world' 'your'
 'yourself' '内容广泛' '内容真实' '内涵' '创意' '制作人' '制作精良' '剪辑能力' '博主是film' '推荐视频'
 '有思想' '有深度' '特点' '用简单却能吸引你看下去的方式讲述道理或现象' '讲故事能力' '高质量']
[[1 0 1 0 0 0 0 1 0 0 0 0 1 0 1 0 0 1 0 0 0 0 2 0 0 1 1 1 0 0 0 0 1 0 0 0
  0 1 0 0 0 0 0 0 0 1 1 0 0 1 0 1 1 1 0 1 0 0 0]
 [0 1 0 1 0 1 0 0 1 1 0 0 0 0 0 0 2 1 0 1 0 1 0 0 0 0 0 0 1 1 0 0 1 1 0 0
  0 2 0 1 0 1 0 0 1 0 0 0 0 0 0 0 1 0 1 1 0 0 1]
 [0 0 0 0 0 0 2 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 1 2 1 0 0 0 0 1 1 0 0 0 1
  0 1 1 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 0 1 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 2 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0
  1 0 3 0 1 0 1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0]]
```



**tf_idf**

```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba

def cut_word(text):
    text = "".join(list(jieba.cut(text)))
    return text

def tf_idf():
    # 获取数据集
    data = [
        'Matt D-Avella  特点：制作精良（博主是film maker） 内容真实  有思想、内涵 推荐视频： 《A Day in the Life of Minimalist》《12 Habits for Life》',
        'Nathaniel Drew 特点：高质量 有深度 内容广泛 推荐视频：《How I Rediscovered Books（In the Digital Age）》《How Learning Languages Changed My View Of the World》',
        'Dan Mace (film maker) 特点：讲故事能力 剪辑能力 创意 推荐视频：《How to edit like Dan Mace》 《The Not Normal Show》',
        '.Johnny Harris  （《Vox-Borders》制作人） 特点：用简单却能吸引你看下去的方式讲述道理或现象 推荐视频：《How to Force Yourself to Learn Stuff》 《How to Remember Your Life》'
        ]

    # 文章分割，即分词
    test_list = []
    for i in data:
        test_list.append(cut_word(i))
    print('分词\n',test_list)

    # 特征提取
    # 实例化转换器
    transfer = TfidfVectorizer()
    new_data = transfer.fit_transform(test_list)

    names = transfer.get_feature_names_out()

    print('特征名称\n', names)
    print(new_data.toarray())

if __name__ == '__main__':
    tf_idf()

# 结果
分词
 ['Matt D-Avella  特点：制作精良（博主是film maker） 内容真实  有思想、内涵 推荐视频： 《A Day in the Life of Minimalist》《12 Habits for Life》', 'Nathaniel Drew 特点：高质量 有深度 内容广泛 推荐视频：《How I Rediscovered Books（In the Digital Age）》《How Learning Languages Changed My View Of the World》', 'Dan Mace (film maker) 特点：讲故事能力 剪辑能力 创意 推荐视频：《How to edit like Dan Mace》 《The Not Normal Show》', '.Johnny Harris  （《Vox-Borders》制作人） 特点：用简单却能吸引你看下去的方式讲述道理或现象 推荐视频：《How to Force Yourself to Learn Stuff》 《How to Remember Your Life》']
特征名称
 ['12' 'age' 'avella' 'books' 'borders' 'changed' 'dan' 'day' 'digital'
 'drew' 'edit' 'film' 'for' 'force' 'habits' 'harris' 'how' 'in' 'johnny'
 'languages' 'learn' 'learning' 'life' 'like' 'mace' 'maker' 'matt'
 'minimalist' 'my' 'nathaniel' 'normal' 'not' 'of' 'rediscovered'
 'remember' 'show' 'stuff' 'the' 'to' 'view' 'vox' 'world' 'your'
 'yourself' '内容广泛' '内容真实' '内涵' '创意' '制作人' '制作精良' '剪辑能力' '博主是film' '推荐视频'
 '有思想' '有深度' '特点' '用简单却能吸引你看下去的方式讲述道理或现象' '讲故事能力' '高质量']
[[0.24040131 0.         0.24040131 0.         0.         0.
  0.         0.24040131 0.         0.         0.         0.
  0.24040131 0.         0.24040131 0.         0.         0.18953516
  0.         0.         0.         0.         0.37907031 0.
  0.         0.18953516 0.24040131 0.24040131 0.         0.
  0.         0.         0.18953516 0.         0.         0.
  0.         0.15344504 0.         0.         0.         0.
  0.         0.         0.         0.24040131 0.24040131 0.
  0.         0.24040131 0.         0.24040131 0.12545138 0.24040131
  0.         0.12545138 0.         0.         0.        ]
 [0.         0.22334394 0.         0.22334394 0.         0.22334394
  0.         0.         0.22334394 0.22334394 0.         0.
  0.         0.         0.         0.         0.28511508 0.17608692
  0.         0.22334394 0.         0.22334394 0.         0.
  0.         0.         0.         0.         0.22334394 0.22334394
  0.         0.         0.17608692 0.22334394 0.         0.
  0.         0.28511508 0.         0.22334394 0.         0.22334394
  0.         0.         0.22334394 0.         0.         0.
  0.         0.         0.         0.         0.11655013 0.
  0.22334394 0.11655013 0.         0.         0.22334394]
 [0.         0.         0.         0.         0.         0.
  0.45172349 0.         0.         0.         0.22586175 0.22586175
  0.         0.         0.         0.         0.14416463 0.
  0.         0.         0.         0.         0.         0.22586175
  0.45172349 0.178072   0.         0.         0.         0.
  0.22586175 0.22586175 0.         0.         0.         0.22586175
  0.         0.14416463 0.178072   0.         0.         0.
  0.         0.         0.         0.         0.         0.22586175
  0.         0.         0.22586175 0.         0.11786403 0.
  0.         0.11786403 0.         0.22586175 0.        ]
 [0.         0.         0.         0.         0.22145689 0.
  0.         0.         0.         0.         0.         0.
  0.         0.22145689 0.         0.22145689 0.28270613 0.
  0.22145689 0.         0.22145689 0.         0.17459916 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.22145689 0.
  0.22145689 0.         0.52379747 0.         0.22145689 0.
  0.22145689 0.22145689 0.         0.         0.         0.
  0.22145689 0.         0.         0.         0.11556539 0.
  0.         0.11556539 0.22145689 0.         0.        ]]
```



