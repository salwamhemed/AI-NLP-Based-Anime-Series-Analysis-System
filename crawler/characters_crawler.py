import scrapy 
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'HxH-Spider'
    start_urls = ['https://hunterxhunter.fandom.com/wiki/List_of_Hunter_×_Hunter_Characters']

    def parse(self, response):
        for href in response.css('.oxy-post-title'):
            extracted_data = scrapy.Request("https://hunterxhunter.fandom.com/wiki/ " +href,
                           callback= self.parse_hxh ) 
            
            yield extracted_data
                           
            yield {'title': href.css('::text').get()}
    
    def parse_hxh(self, response):
        #Getting the character name 
        character_name = response.css("span.mw-page-title-main::text").extract()[0]
        character_name = character_name.strip()
        
        #Getting the character descption
        div_selector = response.css("div.mw-parser-output")
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')
        
        #Getting the character affiliation 
        if soup.find('aside'):
            aside = soup.find('aside')
            affiliation = ''
            for cell in aside.find_all('div', {'class': 'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == 'Affiliation':
                        affiliation = cell.find('div').text.strip()

        soup.find('aside').decompose