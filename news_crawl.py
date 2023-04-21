"""
this code is a Python script that fetches news articles from NewsAPI based on user input for query, 
date range, and language, and writes the retrieved articles to a CSV file. The code performs the 
following steps:

Prompts the user for input for the query (what they want to search for in news articles), date range 
(starting and ending dates for the search), and language (the language in which the articles should 
be retrieved).

Retrieves news articles from NewsAPI using the NewsApiClient from the newsapi library, passing in the 
user input for query, date range, and language as parameters.

Cleans the retrieved article data by removing newlines, extra spaces, and symbols using the clean_text
() function.

Creates a list of lists (indian_news_more) containing the cleaned article data, including URL, title, 
description, and content.

Creates an output directory named "dataset/raw" if it does not exist using the os library.

Writes the retrieved article data to a CSV file named "news_1.csv" in the "dataset/raw" directory 
using the csv library.

Prints a success or error message depending on whether articles were retrieved or not.
"""
import csv
import re
import sys
from colorama import init, Fore
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
news_api_key = os.getenv('NEWS_API')

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running news_crawl.py")

# Initialize NewsApiClient with API key
newsapi = NewsApiClient(api_key=news_api_key)

def clean_text(text):
    """Clean the text by replacing newlines with spaces, removing extra spaces and symbols."""
    if text is not None:
        text = text.replace("\n", " ")
        text = re.sub('\s+', ' ', text).strip()
        text = re.sub('[^0-9a-zA-Z\s]+', '', text)
    return text

def prompt_user(prompt_message, error_message, validation_func):
    """Prompt the user for input, validate the input and return it."""
    while True:
        user_input = input(prompt_message)
        if validation_func(user_input):
            return user_input
        else:
            print(Fore.RED + error_message, file=sys.stderr)

# Prompt user for query, date range and language
query = prompt_user(Fore.YELLOW + "Enter the query you want to search for: ",
                    "Invalid query. Please try again.",
                    lambda x: len(x) > 0)

from_date_str = prompt_user(Fore.YELLOW + "Enter the starting date for the search (YYYY-MM-DD): ",
                            "Invalid date format. Please enter in YYYY-MM-DD format.",
                            lambda x: re.match('\d{4}-\d{2}-\d{2}', x) is not None)

to_date_str = prompt_user(Fore.YELLOW + "Enter the ending date for the search (YYYY-MM-DD): ",
                          "Invalid date format. Please enter in YYYY-MM-DD format.",
                          lambda x: re.match('\d{4}-\d{2}-\d{2}', x) is not None)

language = prompt_user(Fore.YELLOW + "Enter the language you want to search in (e.g. 'en', 'fr', 'es'): ",
                       "Invalid language. Please try again.",
                       lambda x: len(x) == 2)

# Retrieve articles from NewsAPI
articles = newsapi.get_everything(q=query, from_param=from_date_str, to=to_date_str, language=language, sort_by="popularity")

# Create list of lists containing cleaned article data
indian_news_more = [[article["url"], clean_text(article["title"]), clean_text(article["description"]), clean_text(article["content"])] for article in articles["articles"]]

output_dir = r'dataset/raw'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write data to CSV file
with open("dataset/raw/news_1.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["URL", "Title", "Description", "Content"])
    writer.writerows(indian_news_more)

# Log success or error message
if len(indian_news_more) > 0:
    print(Fore.GREEN + "\nSUCCESS: Articles retrieved and written to file dataset/raw/news_1.csv")
else:
    print(Fore.RED + "\nERROR: No articles retrieved.")