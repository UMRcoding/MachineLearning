import csv
import time

import requests
from lxml import etree
import re
import pymysql

# 找目标地址
base_url = 'https://www.hgu.edu.cn/xww/ddyw.htm'
headers ={
            'user-agent': ' ',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-CN;q=0.8,en;q=0.7'
}
# proxies = {
#             'http': 'http://ip:port'
# }
response = requests.get(base_url, headers=headers, timeout=10)
html = etree.HTML(response.content.decode("utf-8"))

# 定位
new_list = html.xpath("/html/body/div[6]/div[2]/div[1]/ul/li")
data_list = []
for news in new_list:
    news_url = news.xpath("./a/@href")[0]
    # 请求详情页
    news_detail_url = "https://www.hgu.edu.cn"+news_url[2:]
    time.sleep(1)
    news_detail_response = requests.get(news_detail_url, headers=headers, timeout=10)
    news_detail_html = etree.HTML(news_detail_response.content.decode("utf-8"))

    # 标题
    news_title = news_detail_html.xpath("/html/body/div[6]/div[2]/form/div/h2/text()")[0]

    # <span style="display:block" class="xh-highlight">发布时间：2022-09-12       &nbsp; &nbsp;来源：党委宣传部
    #            &nbsp; &nbsp;编辑：   </span>
    # <span style="display:block">发布时间：2022-09-12       &nbsp; &nbsp;来源：党委宣传部 &nbsp; &nbsp;编辑：   </span>
    # <span style="display:block">.*?发布时间：(?P<time>.*?) .*?来源：(?P<refer>.*?) .*? </span>
    time_refer_obj = re.compile(r'<span style="display:block">.*?发布时间：(?P<time>.*?) .*?来源：(?P<refer>.*?) .*? </span>',re.S)
    result = time_refer_obj.finditer(news_detail_response.content.decode("utf-8"))
    for item in result:
        news_time = item.group('time')
        news_refer = item.group('refer').strip()

    data = {}
    data["新闻标题"] = news_title
    data["发布时间"] = news_time
    data["新闻链接"] = news_detail_url
    data["新闻来源"] = news_refer
    print(data)

    data_list.append(data)

    # 1）保存CSV
    # 1.1 创建文件对象
    f = open ('data/HGUnews.csv','w', encoding='utf-8-sig', newline='') #newline=''防止空行
    # 1.2 基于对象构建CSV
    csv_write = csv.writer(f)
    # 1.3 构建列表头
    csv_write.writerow(['新闻标题','发布时间','新闻链接','新闻来源'])
    # 1.4 写入CSV文件
    for data in data_list:
        csv_write.writerow([data["新闻标题"],data["发布时间"],data["新闻链接"],data["新闻来源"]])

    # # 保存数据库
    # # 2.1 建立链接
    # conn = pymysql.connect(host='82.156.18.85',port=9909,user='root',password='749034904',db='HGU',connect_timeout=1000)
    # cursor = conn.cursor()
    # # 2.2 创建表
    # sql = '''
    # create table if not exists tb_news(
    #     id int not null auto_increment primary key,
    #     news_time date,
    #     news_url varchar(80),
    #     news_title varchar(100),
    #     news_refer varchar(50)
    #     )
    # '''
    # cursor.execute(sql) #建表
    # # # 2.3 循环插入
    # for data in data_list:
    #     sql = 'insert into tb_news(news_time, news_url, news_title, news_refer) values(%s, %s, %s, %s)'
    #     news_time = data['发布时间']
    #     news_url = data['新闻链接']
    #     news_title = data['新闻标题']
    #     news_refer = data['新闻来源']
    #     cursor.execute(sql, (news_time, news_url, news_title, news_refer))
    #
    # conn.commit()
    # # 2.4 关闭连接
    # conn.close()