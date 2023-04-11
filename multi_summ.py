from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer
from tqdm import tqdm
import csv
import numpy as np
import os

# Load the Pegasus tokenizer and model
model_name = "google/pegasus-xsum"
cache_dir = "cache_dir/transformers"
print('loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('loading model')
model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
print('model loaded')


# Set the input and output directories
input_dir = r'dataset/topics'
output_dir = r'dataset/multi-summaries'

#tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')
#model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')


# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Iterate through all csv files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # Construct the output filename
        output_filename = os.path.splitext(filename)[0] + '.txt'
        # Open the input CSV file
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as csvfile:
            # create a reader object
            reader = csv.reader(csvfile)
            # read the first row, which contains the column names
            column_names = next(reader)
            # find the index of the "full_content" column
            full_content_index = column_names.index("full_content")
            # read the column you want into a list
            my_docs = []
            for row in reader:
                my_docs.append(row[full_content_index])
            # convert the list into a numpy array
            docs = np.array(my_docs)
        
        # Generate summary for each document and export the result summary in a txt file with the same name as the input CSV file
        with open(os.path.join(output_dir, output_filename), "w") as f:
            for doc in tqdm(docs):
                tokens = tokenizer(doc, max_length=128, truncation=True, padding="longest", return_tensors="pt")
                summary = model.generate(**tokens)
                result = tokenizer.decode(summary[0], skip_special_tokens=True)
                f.write(result + "\n")

        print("Summary generation complete for file {}. Check {} for results.".format(filename, output_filename))
