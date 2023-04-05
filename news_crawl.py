import csv
import re
from datetime import datetime, timedelta
from newsapi import NewsApiClient

def clean_text(text):
    if text is not None:
        # Replace newlines with spaces, extra space, symbols
        text = text.replace("\n", " ")
        text = re.sub('\s+', ' ', text).strip()
        text = re.sub('[^0-9a-zA-Z\s]+', '', text)
    return text

newsapi = NewsApiClient(api_key='90501f4112d14afa937413adcde448a3')

# Prompt user to enter query
query = input("Enter the query you want to search for: ")

# Prompt user to enter date range
from_date_str = input("Enter the starting date for the search (YYYY-MM-DD): ")
to_date_str = input("Enter the ending date for the search (YYYY-MM-DD): ")

# Prompt user to enter language
language = input("Enter the language you want to search in (e.g. 'en', 'fr', 'es'): ")

articles = newsapi.get_everything(q=query, from_param=from_date_str, to=to_date_str, language=language, sort_by="popularity")

indian_news_more = [[article["url"], clean_text(article["title"]), clean_text(article["description"]), clean_text(article["content"])] for article in articles["articles"]]

with open("dataset/indian_news_more.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["URL", "Title", "Description", "Content"])
    writer.writerows(indian_news_more)