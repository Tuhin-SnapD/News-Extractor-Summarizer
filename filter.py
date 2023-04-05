import os
import pandas as pd

news_data = pd.read_csv('dataset/indian_news_more_with_content_sd_topic.csv')

# Get a list of all unique topics
topics = news_data['topic'].unique()

# Function to create separate CSV files for each topic
def create_topic_csv(topic_name, data):
    topic_rows = data[data['topic'] == topic_name]
    topic_rows = topic_rows.dropna()  # Drop rows with missing values
    if not topic_rows.empty:
        folder_name = 'dataset_for_md_summ'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        filename = f'{topic_name}.csv'
        file_path = os.path.join(folder_name, filename)
        topic_rows.to_csv(file_path, index=False)

# Iterate through the topics and create separate CSV files
for topic in topics:
    create_topic_csv(topic, news_data)