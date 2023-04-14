"""
In general, this code is a Python script that reads a CSV file containing news data with full content 
from a file named 'dataset/raw/news_with_full_content_2.csv' using the pandas library. It then 
updates the values in the "final_full_content" column by removing a specific pattern (indicated by '\d
{4} chars$') using regular expressions. 

The updated DataFrame is then written back to the same CSV 
file, overwriting the original data, using the to_csv() method with the parameter index=False to 
exclude index column in the output file. The colorama library is used to print a yellow-colored 
message indicating that the script is running. The autoreset=True parameter in the init() function is 
used to automatically reset the text color after each print statement.
"""
import pandas as pd
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running data_clean.py")

# Load the CSV file
df = pd.read_csv('dataset/raw/news_with_full_content_2.csv')

# Update the "final_full_content" column
df['final_full_content'] = df['final_full_content'].str.replace(r'\d{4} chars$', '', regex=True)
# Save the updated DataFrame back to the same CSV file
df.to_csv('dataset/raw/news_with_full_content_2.csv', index=False)

print(Fore.GREEN + "\nData has been cleaned present in dataset/raw/news_with_full_content_2.csv")

