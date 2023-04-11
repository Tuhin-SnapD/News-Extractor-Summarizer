import os
import pandas as pd
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

news_data = pd.read_csv(r'dataset\raw\news_with_full_content_with_topic_3.csv')

# Get a list of all unique topics
topics = news_data['topic'].unique()

import os
import re

# Function to create separate CSV files for each topic
def create_topic_csv(topic_name, data):
    topic_rows = data[data['topic'] == topic_name]
    topic_rows = topic_rows.dropna()  # Drop rows with missing values
    if not topic_rows.empty:
        folder_name = r'dataset/topics'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        filename = f'{topic_name}.csv'
        file_path = os.path.join(folder_name, filename)
        topic_rows.to_csv(file_path, index=False)
        
        # # Extract the "one line summary" column and export to a text file
        # summary_col = topic_rows['one line summary']
        # summary_folder = 'summaries'
        # if not os.path.exists(summary_folder):
        #     os.mkdir(summary_folder)
        # summary_file_path = os.path.join(summary_folder, f'{topic_name}_one_line_summary.txt')
        # with open(summary_file_path, 'w') as f:
        #     for summary in summary_col:
        #         # Add full stop after each sentence
        #         summary = re.sub(r'(?<=[a-z0-9][.?!]) +(?=[a-zA-Z])', '. ', summary.strip())
        #         f.write(summary + '\n')

# Iterate through the topics and create separate CSV files
for topic in topics:
    create_topic_csv(topic, news_data)


print(Fore.GREEN + "\nSUCCESS: Check dataset/topics")