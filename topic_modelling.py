from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# Load the tokenizer and model
model_name = "mrm8488/t5-base-finetuned-news-title-classification"
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-news-title-classification", use_auth_token='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')
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-news-title-classification", use_auth_token='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')

def topic(article_text):
    try:
        input_ids = tokenizer.encode(
            article_text, return_tensors='pt').to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
        topic_name = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except:
        topic_name = "Unable to generate topic"
    return topic_name


input_file_path = '/content/indian_news_more_with_content_sd.csv'
output_file_path = '/content/indian_news_more_with_content_sd_topic.csv'

# read the input CSV file into a pandas data frame
with open(input_file_path) as f:
    df = pd.read_csv(f)

# add a new column for the topic
df['topic'] = ''

# iterate over each row in the data frame and add the topics
df['topic'] = [topic(article_text) for article_text in df['one line summary']]

# write the updated data frame to a new CSV file
with open(output_file_path, 'w', newline='') as f:
    df.to_csv(f, index=False)