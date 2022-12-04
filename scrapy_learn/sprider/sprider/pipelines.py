# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import json

from itemadapter import ItemAdapter


# 存储
class SpriderPipeline:
    def __init__(self):
        self.file = open("hgu.json", "wb+")
    def process_item(self, item, spider):
        # 入库
        str_data = json.dumps(dict(item),ensure_ascii=False)+',\n'
        self.file.write(str_data.encode("utf-8"))
        return item
    def __del__(self):
        self.file.close()