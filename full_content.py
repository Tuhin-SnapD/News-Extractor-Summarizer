"""
The code is a Python script that fetches article content from URLs listed in a CSV file, filters and 
processes the content using regular expressions and spaCy library, and then writes the processed 
content back to the CSV file in a new column. Here's an overview of what the code does:

Imports necessary libraries including pandas, newspaper, requests, re, tqdm, colorama, and spacy.
Defines a function called find_final_url(url) that takes a URL as input and retrieves the final URL 
after any redirections.

Defines a function called get_article_content(url) that takes a URL as input, downloads the article 
content from the URL using the newspaper library, and parses the content to extract the text.

Defines a function called filter_content(text) that takes the extracted text as input, processes it 
using spaCy library to extract main content by selecting a fixed number of sentences (4 in this 
case), and then filters out unwanted characters and whitespaces using regular expressions.

Defines input and output file paths for the CSV file that contains the URLs.

Reads the input CSV file into a pandas DataFrame.

Adds two new columns to the DataFrame for storing the full article content and the final article 
content after processing.

Iterates over each row in the DataFrame and for each URL, retrieves the final URL, fetches the 
article content using the get_article_content(url) function, and filters the content using the 
filter_content(text) function.

Performs multiple regex substitutions on the filtered content to remove unwanted characters and 
whitespaces.

Stores the filtered content in the 'full_content' column and selects the larger text between the 
filtered content, 'Description', and 'Content' columns as the final content, and stores it in the 
'final_full_content' column.

Writes the updated DataFrame to a new CSV file.

Prints a message indicating that the process is complete and the new CSV file is generated.
"""
import pandas as pd
import newspaper
from newspaper import Config
import requests
import re
from tqdm import tqdm
from colorama import init, Fore

import spacy
# Load the English language model
nlp = spacy.load("en_core_web_sm")

print(Fore.YELLOW + "Running full_content.py")

# Initialize colorama
init(autoreset=True)

# function to extract final URL
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
        print(Fore.RED + f"\nError fetching content for URL: {url}")
        print(Fore.RED + str(e))
        return ''
    
def filter_content(text):

    # Set the number of sentences for the main content
    num_sentences = 4
    # Process the text with spaCy
    doc = nlp(text)

    # Extract the sentences
    sentences = [sent.text for sent in doc.sents]
    # Get the first 'num_sentences' sentences as main content
    main_content = " ".join(sentences[:num_sentences])

    # Filter out the main content from the original text
    filtered_text = main_content
    # Filter out the main content from the original text
    filtered_text = "".join([str(sentence) for sentence in main_content])
    return filtered_text

input_file = r'dataset/raw/news_1.csv'
output_file = r'dataset/raw/news_with_full_content_2.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file)

# add a new column for the article content
df['raw_full_content'] = ''
df['spacy_full_content'] = ''
df['final_full_content']=''

# iterate over rows and get content for each URL
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching article content"):
    url = row['URL']
    url = find_final_url(url)
    content = get_article_content(url)
    raw_content = content
    content = filter_content(content)

    # Perform multiple regex substitutions on the 'content' variable:
    # 1. Replace one or more consecutive whitespaces with a single space
    # 2. Remove any characters that are not alphanumeric or whitespaces
    # 3. Remove any characters that are not word characters, whitespaces, or periods
    # Finally, strip any leading or trailing spaces from the resulting string
    raw_content = re.sub(r'\s+', ' ', raw_content).strip()
    raw_content = re.sub(r'[^0-9a-zA-Z\s.?!]+', '', raw_content)

    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'[^0-9a-zA-Z\s.?!]+', '', content)
    content = re.sub(r'[^\w\s.?!]', '', content)

    text = content
    df.loc[idx, 'raw_full_content'] = raw_content
    df.loc[idx, 'spacy_full_content'] = text
    

    larger_text = max([text, row['Description'], row['Content']], key=len)
    larger_text = re.sub(r'\d{4} chars$', '', larger_text)
    df.loc[idx, 'final_full_content'] = larger_text

# write the updated data frame to a new CSV file
df.to_csv(output_file, index=False)

print(Fore.GREEN + "\nAll full contents have been fetched, check dataset/raw/news_with_full_content_2.csv")
