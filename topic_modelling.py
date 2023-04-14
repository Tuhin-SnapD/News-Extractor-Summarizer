"""
This code is a Python script that uses the Hugging Face Transformers library to perform topic 
modeling on a CSV file containing news data with titles. The code performs the following steps:

Imports necessary libraries including transformers, pandas, tqdm, colorama, and os.

Initializes colorama to auto-reset text color after each print statement.

Loads a pre-trained Seq2Seq language model for news title classification using the "mrm8488/
t5-base-finetuned-news-title-classification" model from the Transformers library.

Defines a function named "topic" that takes text as input, encodes it using the tokenizer, generates 
topic using the pre-trained model, and decodes the output to get the predicted topic.

Specifies the input and output file paths for the CSV files containing the news data.

Reads the input CSV file into a pandas DataFrame.

Adds a new column named "topic" to the DataFrame to store the predicted topics.

Iterates over each row in the DataFrame using tqdm (a progress bar library) and calls the "topic" 
function to fetch the predicted topic for each title. The predicted topic is stored in the "topic" 
column of the DataFrame.

Writes the updated DataFrame to a new CSV file without row indices.

Prints a green-colored message indicating that the topics have been generated and the output file path where the updated data is saved.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from colorama import init, Fore
import os

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running topic_modelling.py")

# Load the tokenizer and model
print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('cache_dir/transformers/mrm8488/t5-base-finetuned-news-title-classification')
print('Loading model')
model = T5ForConditionalGeneration.from_pretrained('cache_dir/transformers/mrm8488/t5-base-finetuned-news-title-classification')
print('Model and Tokenizer loaded')

def topic(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    outputs = model.generate(input_ids=input_ids, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


input_file_path = r'dataset/raw/news_with_full_content_2.csv'
output_file_path = r'dataset/raw/news_with_full_content_with_topic_3.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file_path)

# add a new column for the article topic
df['topic'] = ''

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching article topic"):
    text = row['Title']
    topic_title = topic(text)
    df.loc[idx, 'topic'] = topic_title

# write the updated data frame to a new CSV file without row indices
df.to_csv(output_file_path, index=False)

print(Fore.GREEN + "\nAll topics have been generated, check dataset/raw/news_with_full_content_with_topic_3.csv")
