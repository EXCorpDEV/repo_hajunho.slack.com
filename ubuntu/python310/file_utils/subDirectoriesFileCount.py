import os

directory = "./"

file_counts = {}
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):
        file_counts[folder] = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

for folder, count in file_counts.items():
    print(f"{folder}: {count} files")