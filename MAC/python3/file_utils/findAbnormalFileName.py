import os
import re

def find_and_log_non_standard_filenames(directory):
    # Regular expression to match non-standard characters in file names
    # This pattern matches any character that is not a letter, a number, an underscore, or a period
    pattern = r'[^\w.\-]'

    # List to hold the names of files with non-standard characters
    non_standard_filenames = []

    # Listing all files in the provided directory
    for filename in os.listdir(directory):
        print(f"Checking file: {filename}")  # Logging the file being checked
        # Search for non-standard characters in the file name
        if re.search(pattern, filename):
            non_standard_filenames.append(filename)

    return non_standard_filenames

# Replace '.' with the path to the directory containing the files you want to check.
directory_path = '.' # Placeholder for the directory path
# Call the function and print the non-standard file names
non_standard_filenames = find_and_log_non_standard_filenames(directory_path)
print(f"list : {non_standard_filenames}")