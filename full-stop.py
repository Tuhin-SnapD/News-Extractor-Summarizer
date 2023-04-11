import os

# Directory path where the txt files are located
directory = r"dataset/multi-summaries"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a txt file
    if filename.endswith(".txt"):
        # Construct the full file path by joining the directory and filename
        filepath = os.path.join(directory, filename)

        # Open the file in read mode and read all lines
        with open(filepath, 'r+') as file:
            lines = file.readlines()

            # Move the file pointer to the beginning of the file
            file.seek(0)

            # Iterate through each line
            for line in lines:
                # Strip leading and trailing whitespaces from the line
                line = line.strip()

                # Check if the line ends with a period
                if not line.endswith('.'):
                    # If not, add a period at the end of the line
                    line += '.'

                # Write the updated line to the file
                file.write(line + '\n')

            # Truncate the file to the current position of the file pointer
            file.truncate()
