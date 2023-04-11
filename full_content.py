import pandas as pd
import newspaper
from newspaper import Config
import requests
import re
from tqdm import tqdm
import logging

# configure logging
logging.basicConfig(level=logging.INFO)

# function to extract final URL from consent.google.com URLs
def find_final_url(url):
    try:
        response = requests.get(url)
        final_url = response.url
        return final_url
    except requests.exceptions.RequestException as e:
        return url

# function to get the article content from a URL
def get_article_content(url):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    article = newspaper.Article(url, config=config)
    try:
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Error fetching content for URL: {url}")
        logging.error(str(e))
        return ''

input_file = r'dataset/raw/news_1.csv'
output_file = r'dataset/raw/news_with_full_content_2.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file)

# add a new column for the article content
df['full_content'] = ''

# iterate over rows and get content for each URL
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching article content"):
    url = row['URL']
    url = find_final_url(url)
    content = get_article_content(url)

    # Remove extra spaces and symbols
    content = re.sub('\s+', ' ', content).strip()
    content = re.sub('[^0-9a-zA-Z\s]+', '', content)

    if content:
        df.loc[idx, 'full_content'] = content
    else:
        df.loc[idx, 'full_content'] = row['Content']

# write the updated data frame to a new CSV file
df.to_csv(output_file, index=False)

logging.info(f"All full contents have been fetched")