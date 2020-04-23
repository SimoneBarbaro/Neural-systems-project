import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

def join_p_with_symbol(body):
    p_list = []
    for p in body.find_all('p'):
        p_list.append(p.text)
    body_str = '<--->'.join(p_list)
    return body_str

# =============================================================================
# write web collected results and discussion contents with collected ids into csv file
# =============================================================================
source_location_content_ids = './data/articleList0_web_ids.csv'
target_location_content     = './data/articleList0_RnD.csv'

article_list = pd.read_csv(source_location_content_ids, encoding = "ISO-8859-1")
urls = article_list["WEB"]
R_ids = article_list["Result_ID_WEB"]
D_ids = article_list["Discussion_ID_WEB"]

# extract RnD content from website with extracted url and ids
for i in range(0,len(urls)):
    url  = urls[i]
    R_id = R_ids[i]
    D_id = D_ids[i]
    content = requests.get(url).text
    soup = BeautifulSoup(content, 'lxml')
    Result_Content_soup = soup.find('div', id=R_id, class_="tsec sec")  # .text
    Result_Content = join_p_with_symbol(Result_Content_soup)
    Discussion_Content_soup = soup.find('div', id=D_id, class_="tsec sec")
    Discussion_Content = join_p_with_symbol(Discussion_Content_soup)
    # write the content into dataframe
    article_list["Result_Content"][i] = Result_Content
    article_list["Discussion_Content"][i] = Discussion_Content

article_list.to_csv(target_location_content)