import requests
import re
import time
from bs4 import BeautifulSoup
from lxml.html import fromstring

FIELDS = ('national_flag', 'area', 'population', 'iso', 'country', 'capital', 'continent', 'tld',
          'currency_code', 'currency_name', 'phone', 'postal_code_format',
          'postal_code_regex', 'languages', 'neighbours')
NUM_ITERATION = 1000


def re_scraper(html):
    results = {}
    for field in FIELDS:
        match = re.search('<tr id="places_%s__row">.*?<td class="w2p_fw">(.*?)</td>' % field, html)
        if match is None:
            continue
        results[field] = match.groups()[0]
    return results


def bs_scraper(html):
    soup = BeautifulSoup(html, 'lxml')
    results = {}
    for field in FIELDS:
        tr = soup.find('table').find('tr', id='places_%s__row' % field)
        if tr is None:
            continue
        results[field] = tr.find('td', class_='w2p_fw').text
    return results


def lxml_scraper(html):
    tree = fromstring(html)
    results = {}
    for field in FIELDS:
        cssselect = tree.cssselect('table > tr#places_%s__row > td.w2p_fw' % field)
        if cssselect is None and len(cssselect) < 1:
            continue
        results[field] = cssselect[0].text_content()
    return results


def lxml_xpath_scraper(html):
    tree = fromstring(html)
    results = {}
    for field in FIELDS:
        xpath = tree.xpath("//tr[@id='places_%s__row']/td[@class='w2p_fw']" % field)
        if xpath is None:
            continue
        results[field] = xpath[0].text_content()
    return results


scrapers = [
    ('Regular Expression', re_scraper),
    ('BeautifulSoup', bs_scraper),
    ('Lxml', lxml_scraper),
    ('Xpath', lxml_xpath_scraper)
]

html = requests.get("http://example.webscraping.com/places/default/view/Aland-Islands-2").text

for name, scraper in scrapers:
    start = time.time()
    for i in range(NUM_ITERATION):
        result = scraper(html)
    end = time.time()
    print('%s: %.2f seconds' % (name, end-start))