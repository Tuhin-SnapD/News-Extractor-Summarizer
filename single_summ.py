from transformers import PegasusForConditionalGeneration, AutoTokenizer
import pandas as pd

model_name = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def summarise(text):
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary = model.generate(**tokens,  max_new_tokens=60)
    summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
    return summary_text

input_file_path = 'dataset/indian_news_more_with_content.csv'
output_file_path = 'dataset/indian_news_more_with_content_sd.csv'

# read the input CSV file into a pandas data frame
with open(input_file_path) as f:
    df = pd.read_csv(f)

# add a new column for the summary
df['one line summary'] = ''

# iterate over each row in the data frame and add the summaries
df['one line summary'] = [summarise(text) for text in df['content']]

# write the updated data frame to a new CSV file
with open(output_file_path, 'w', newline='') as f:
    df.to_csv(f, index=False)
