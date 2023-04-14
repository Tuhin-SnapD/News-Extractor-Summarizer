"""
In general, this code is a Python script that uses the Pegasus model from the Transformers library to 
generate summaries for a collection of documents stored in multiple CSV files. The code performs the 
following steps:

Imports necessary libraries including Transformers (specifically, PegasusForConditionalGeneration, 
PegasusTokenizer, and AutoTokenizer), tqdm, csv, numpy, os, and colorama (for colored console output).

Initializes colorama to auto-reset text color after each print statement.

Prints a yellow-colored message indicating that the script is running.

Loads the Pegasus tokenizer and model using the model name "google/pegasus-xsum" and caches the model 
in a directory named "cache_dir/transformers".

Sets the input and output directories where the CSV files with documents and generated summaries will 
be read from and written to, respectively.

Creates the output directory if it does not exist.

Iterates through all CSV files in the input directory.

For each CSV file, it opens the file and reads the "full_content" column into a list named "my_docs".

Converts the list "my_docs" into a numpy array named "docs".

Generates summaries for each document in "docs" using the loaded Pegasus model and writes the 
generated summary to a text file with the same name as the input CSV file but with a ".txt" extension 
in the output directory.

Prints a green-colored message indicating that the summary generation is completed and provides the 
output folder path where the generated summaries are saved.
"""

from transformers import PegasusForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import csv
import numpy as np
import os
from colorama import init, Fore
import re

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running multi_summ.py")

# Load the tokenizer and model
print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('cache_dir/transformers/google/xsum')
print('Loading model')
model = PegasusForConditionalGeneration.from_pretrained('cache_dir/transformers/google/xsum')
print('Model and Tokenizer loaded')


# Set the input and output directories
input_dir = r'dataset/topics'
output_dir = r'dataset/multi-summaries'

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
            full_content_index = column_names.index("final_full_content")
            # read the column you want into a list
            my_docs = []
            for row in reader:
                my_docs.append(row[full_content_index])
            # convert the list into a numpy array
            docs = np.array(my_docs)
        
        # Generate summary for each document and export the result summary in a txt file with the same name as the input CSV file
        with open(os.path.join(output_dir, output_filename), "w") as f:
            for doc in tqdm(docs, desc="Summarising articles"):
                text = re.sub(r'[^\w\s\.]', '', doc)
                tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
                summary = model.generate(**tokens, max_new_tokens=128) # Update max_length to max_new_tokens
                result = tokenizer.decode(summary[0], skip_special_tokens=True)
                f.write(result + "\n")
        print(Fore.GREEN + "Summary generation completed, file {}. Check {} for results.\n".format(filename, output_filename))