import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


## get url from PMC by given DOIs
def get_PMC_url(DOI):
    url = "https://www.ncbi.nlm.nih.gov/pmc/?term=" + DOI
    content = requests.get(url).text
    bsObj = BeautifulSoup(content, 'lxml')
    t1 = bsObj.find('dd').text
    t2 = "https://www.ncbi.nlm.nih.gov/pmc/articles/" + t1
    return t2

## find article weblink
def DOIs_find_urls(location):
    list_article = pd.read_csv(location, encoding = "ISO-8859-1")
    DOI = list_article["DOI"]
    count = 0
    for doi in DOI:
        url = get_PMC_url(doi)
        list_article["WEB"][count] = url
        count += 1
    return list_article
    

# =============================================================================
# # obtaining article's website according to article's DOI
# =============================================================================
location1_DOIs = './data/articleList1_DOIs.csv'
location2_DOIs = './data/articleList2_DOIs.csv'
df_with_urls1 = DOIs_find_urls(location1_DOIs)
df_with_urls2 = DOIs_find_urls(location2_DOIs)

location1_URLs = './data/articleList1_DOIs_urls.csv'
location2_URLs = './data/articleList2_DOIs_urls.csv'
df_with_urls1.to_csv(location1_URLs)
df_with_urls2.to_csv(location2_URLs)