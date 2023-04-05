import pandas as pd
import newspaper
from newspaper import Config
import requests
from urllib.parse import urlparse
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
    except:
        return ''

input_file = 'dataset/indian_news_more.csv'
output_file = 'dataset/indian_news_more_with_content.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file)

# add a new column for the article content
df['full-content'] = ''

# iterate over each row in the input CSV file
for i, row in enumerate(df.itertuples(), start=1):
    url = row[1]
    url = find_final_url(url)
    content = get_article_content(url)

    # Remove extra spaces and symbols
    content = re.sub('\s+', ' ', content).strip()
    content = re.sub('[^0-9a-zA-Z\s]+', '', content)

    if content:
        print(f"successfully fetched content for row {i}")
        df.at[i-1, 'full-content'] = content
    else:
        print(f"Error fetching content for row {i}, using default description")
        print(url)
        df.at[i-1, 'full-content'] = row[4]

# write the updated data frame to a new CSV file
df.to_csv(output_file, index=False)