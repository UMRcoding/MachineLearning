# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

# 解析模板
class SpriderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapydemo.Field()
    title = scrapy.Field()
    url = scrapy.Field()
    date_publish = scrapy.Field()

    content = scrapy.Field()
