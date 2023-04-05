import pandas as pd
import newspaper
from newspaper import Config
import requests
import re

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
        print(f"Error fetching content for URL: {url}")
        print(str(e))
        return ''


input_file = 'dataset/indian_news_more.csv'
output_file = 'dataset/indian_news_more_with_content.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file)

# add a new column for the article content
df['full_content'] = ''

# extract final URLs and get article content for each row
for idx, row in df.iterrows():
    url = row['url']
    url = find_final_url(url)
    content = get_article_content(url)

    # Remove extra spaces and symbols
    content = re.sub('\s+', ' ', content).strip()
    content = re.sub('[^0-9a-zA-Z\s]+', '', content)

    if content:
        print(f"successfully fetched content for URL: {url}")
        df.loc[idx, 'full_content'] = content
    else:
        print(f"Using default description for URL: {url}")
        df.loc[idx, 'full_content'] = row['description']

# write the updated data frame to a new CSV file
df.to_csv(output_file, index=False)
