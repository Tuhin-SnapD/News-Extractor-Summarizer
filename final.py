"""
This code is designed to display the contents of PNG files and their corresponding TXT files in a 
specified directory. Here's an overview of what the code does:

Imports the necessary libraries, PIL (Pillow) for image processing and os for file system operations.

Defines the directory where the PNG and TXT files are located using the variable directory.

Loops through all the files in the specified directory using os.listdir(directory).

Checks if each file has a .png extension using filename.endswith('.png').

Extracts the file name without extension from the PNG file name using os.path.splitext(filename)[0] 
and stores it in the variable file_name_without_ext.

Constructs the file path for the corresponding TXT file by appending the file_name_without_ext with .
txt and prepending it with the directory where the TXT files are located, which is 'dataset/final'. 
This is done using os.path.join('dataset/final', file_name_without_ext + '.txt') and stored in the 
variable txt_file_path.

Checks if the corresponding TXT file exists using os.path.exists(txt_file_path).

If the TXT file exists, it opens and displays the PNG file using Image.open(png_file_path) and img.
show(), where png_file_path is the complete file path of the PNG file constructed by joining the 
directory with the filename.

It then reads and displays the contents of the TXT file using with open(txt_file_path, 'r') as 
txt_file and txt_file.read(). The file name of the TXT file (without extension) is also displayed 
along with its contents using print("Contents of {}:".format(file_name_without_ext + '.txt')).

If the corresponding TXT file does not exist, it prints a message indicating that no corresponding 
TXT file was found for the PNG file using print("No corresponding TXT file found for {}".format
(filename)).
"""
from PIL import Image
import os
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)

print(Fore.YELLOW + "Running final.py")

# Define the directory where the PNG and TXT files are located
directory = 'dataset/graphs'

# Loop through all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        # Extract the file name without extension
        file_name_without_ext = os.path.splitext(filename)[0]
        
        # Check if there is a corresponding TXT file
        txt_file_path = os.path.join('dataset/final', file_name_without_ext + '.txt')
        if os.path.exists(txt_file_path):
            # Open and display the PNG file
            png_file_path = os.path.join(directory, filename)
            img = Image.open(png_file_path)
            img.show()
            
            # Read and display the contents of the TXT file along with the file name
            with open(txt_file_path, 'r') as txt_file:
                txt_contents = txt_file.read()
                print(Fore.GREEN + "Most important news of {}:".format(file_name_without_ext))
                print(txt_contents)
                print()
        else:
            print("No corresponding TXT file found for {}".format(filename))
