import scrapy

from sprider.items import SpriderItem

class HguSpider(scrapy.Spider):
    name = 'hgu'
    # 允许访问的域名
    allowed_domains = ['hgu.edu.cn']
    start_urls = ['https://www.hgu.edu.cn/xww/ddyw/111.htm']
    curpage = 111

    # url -> 引擎 -> 调度器 -> 引擎 -> 下载器 -> 下载（请求） -> 相应的结果 response -> 引擎 -> spider
    def parse(self, response):
        self.curpage = self.curpage - 1
        node_list = response.xpath("//div[@class='text_list']/ul/li")
        for node in node_list:
            item = SpriderItem()
            item['title'] = node.xpath('.//b/text()').extract()
            item['url'] = node.xpath('./a/@href').extract()
            item['date_publish'] = node.xpath('.//i/text()').extract()
            # yield item  # 提交到pipeline

            news_detail = item['url']
            news_detail_url = "https://www.hgu.edu.cn" + news_detail[0][5:]
        #   构造详情request
            yield scrapy.Request(
                url = news_detail_url,
                callback = self.parse_detail,
                meta = {'item':item}  # 用yield发送 item  # 携带meta 提交到pipeline
            )

        # 翻页模拟
        curpages = self.curpage
        if self.curpage >= 1:
            next_url = f'https://www.hgu.edu.cn/xww/ddyw/{curpages}.htm'
            # 构建请求对象，并且返回给引擎，回调 自己好了就交付给别人，最后yield给引擎
            yield scrapy.Request(
                url = next_url,
                callback = self.parse
            )

    def parse_detail(self, response):
        item = response.meta['item']
        item['content'] = response.xpath("/html/body/div[6]/div[2]/form/div/div[1]/div[@class='v_news_content']").extract()  # .extract() 抽取内容
        yield item