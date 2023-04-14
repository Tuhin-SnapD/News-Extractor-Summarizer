"""
In general, this code is a Python script that processes a CSV file containing news data with topics 
and creates separate CSV files for each unique topic. The code performs the following steps:

Imports necessary libraries including os, pandas, colorama, and init from colorama.

Initializes colorama to auto-reset text color after each print statement.

Reads a CSV file containing news data into a pandas DataFrame named "news_data".

Gets a list of all unique topics from the "topic" column of the "news_data" DataFrame.

Defines a function named "create_topic_csv" that takes a topic name and data as input. Within the 
function, it filters the rows of "news_data" DataFrame to get the rows with the specified topic, 
drops rows with missing values, creates a folder named "dataset/topics" if it doesn't exist, and 
saves the filtered data to a CSV file with the topic name as the file name in the "dataset/topics" 
folder.

Iterates through the unique topics obtained in step 4, and calls the "create_topic_csv" function for 
each topic to create separate CSV files for each topic.

Prints a green-colored message indicating that the topic-specific CSV files have been created and 
provides the output folder path where the files are saved.
"""
import os
import pandas as pd
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running topic_cluster_formation.py")
# Read the CSV file
news_data = pd.read_csv(r'dataset\raw\news_with_full_content_with_topic_3.csv')

# Get a list of all unique topics
topics = news_data['topic'].unique()

# Function to create separate CSV files for each topic
def create_topic_csv(topic_name, data):
    topic_rows = data[data['topic'] == topic_name].dropna()  # Drop rows with missing values
    if not topic_rows.empty:
        folder_name = r'dataset/topics'
        os.makedirs(folder_name, exist_ok=True)  # Create directory if it doesn't exist
        filename = f'{topic_name}.csv'
        file_path = os.path.join(folder_name, filename)
        topic_rows.to_csv(file_path, index=False)

# Iterate through the topics and create separate CSV files
for topic in topics:
    create_topic_csv(topic, news_data)

print(Fore.GREEN + "\nTopic clusters have been created, check dataset/topics")