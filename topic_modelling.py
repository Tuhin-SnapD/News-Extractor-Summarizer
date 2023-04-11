from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import csv

# # Load the tokenizer and model
# model_name = "mrm8488/t5-base-finetuned-news-title-classification"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Set the path of the cache directory
#cache_dir = "cache_dir_2/transformers"

# Load the tokenizer and model
model_name = r"cache_dir_2/transformers/mrm8488_new/t5-base-finetuned-news-title-classification_new"
print('loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('loading model')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print('model loaded')





def topic(text):
  input_ids = tokenizer.encode(title, return_tensors='pt').to(model.device)
  outputs = model.generate(input_ids=input_ids)
  output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return output_text


input_file_path = r'dataset/raw/news_with_full_content_2.csv'
output_file_path = r'dataset/raw/news_with_full_content_with_topic_3.csv'

rows_with_topic = []
count = 1


# open the input CSV file and read each row
with open(input_file_path, 'r') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    header.append('topic')
    # iterate over each row in the data frame and add the topics
    for row in reader:
        title = row[2]
        row.append(topic(title))
        print(f"Completed for row {count}")
        count = count + 1
        rows_with_topic.append(row)

# write the updated data frame to a new CSV file
with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)

    for row in rows_with_topic:
        writer.writerow(row)