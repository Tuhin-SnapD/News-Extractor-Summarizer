import csv
from transformers import PegasusForConditionalGeneration, AutoTokenizer
import pandas as pd

model_name = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def summarise(text):
  tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
  summary = model.generate(**tokens,  max_new_tokens=60)
  str = tokenizer.decode(summary[0], skip_special_tokens=True)
  return str

input_file_path = 'dataset/indian_news_more_with_content.csv'
output_file_path = 'dataset/indian_news_more_with_content_sd.csv'

# read the input CSV file into a pandas data frame
df = pd.read_csv(input_file_path)

# add a new column for the summary
df['one line summary'] = ''

# iterate over each row in the data frame
for i, row in enumerate(df.itertuples(), start=1):
    text = row[5]
    one_line_summary = summarise(text)
    df.at[i-1, 'one line summary'] = one_line_summary
    print(f"Success for {i}")

# write the updated data frame to a new CSV file
df.to_csv(output_file_path, index=False)

# df.head(10)

# row[5]

# df.tail(10)

