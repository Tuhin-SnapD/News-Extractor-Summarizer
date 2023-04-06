import csv
import re
from newsapi import NewsApiClient


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
            print(error_message)


# Initialize NewsApiClient with API key
newsapi = NewsApiClient(api_key='90501f4112d14afa937413adcde448a3')

# Prompt user for query, date range and language
query = prompt_user("Enter the query you want to search for: ",
                    "Invalid query. Please try again.",
                    lambda x: len(x) > 0)

from_date_str = prompt_user("Enter the starting date for the search (YYYY-MM-DD): ",
                            "Invalid date format. Please enter in YYYY-MM-DD format.",
                            lambda x: re.match('\d{4}-\d{2}-\d{2}', x) is not None)

to_date_str = prompt_user("Enter the ending date for the search (YYYY-MM-DD): ",
                          "Invalid date format. Please enter in YYYY-MM-DD format.",
                          lambda x: re.match('\d{4}-\d{2}-\d{2}', x) is not None)

language = prompt_user("Enter the language you want to search in (e.g. 'en', 'fr', 'es'): ",
                       "Invalid language. Please try again.",
                       lambda x: len(x) == 2)

try:
    # Retrieve articles from NewsAPI
    articles = newsapi.get_everything(
        q=query, from_param=from_date_str, to=to_date_str, language=language, sort_by="popularity")

    # Create list of lists containing cleaned article data
    indian_news_more = [[article["url"], clean_text(article["title"]), clean_text(
        article["description"]), clean_text(article["content"])] for article in articles["articles"]]

    # Write data to CSV file
    with open("dataset/news_1.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Title", "Description", "Content"])
        writer.writerows(indian_news_more)

    print(f"{len(indian_news_more)} articles written to file.")

except Exception as e:
    print(f"An error occurred while retrieving articles: {e}")
