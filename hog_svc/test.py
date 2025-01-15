import os

folder_path = '/home/gustavo/Documents/ufmg/2024_2/introducao_a_computacao_visual/tp_02/hog_svc/labels10k'  # Replace with your folder path

# List all files in the folder
files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

for file in files:
    file_path = os.path.join(folder_path, file)
    
    with open(file_path, 'r') as f:
        content = f.read().replace(' ', '').replace('\n', '')  # Remove spaces and newlines
        
        if len(content) != 6:
            print(f'File "{file}" has more than 6 non-space characters.')