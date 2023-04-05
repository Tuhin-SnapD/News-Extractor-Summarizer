# !pip install sentencepiece
# !pip install --upgrade tokenizer
# !pip install transformers tokenizers
# !pip install torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

# Load the tokenizer and model
model_name = "mrm8488/t5-base-finetuned-news-title-classification"
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-news-titles-classification")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-news-titles-classification")

def topic(text):
  input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
  outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
  output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return output_text

input_file_path = '/content/indian_news_more_with_content_sd.csv'
output_file_path = '/content/indian_news_more_with_content_sd_topic.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file_path)

# add a new column for the summary
df['topic'] = ''

# iterate over each row in the data frame
for i, row in enumerate(df.itertuples(), start=1):
    text = row[6]
    topic_name = topic(text)
    df.at[i-1, 'topic'] = topic_name
    print(f"Success for {i}")

df.to_csv(output_file_path, index=False)

# df.tail(20)

# row[6]

