# import os
# import pandas as pd
# import random
# import csv

# # def get_random_row_from_csv(csv_files):
# #     # Increase the field size limit to handle larger fields
# #     max_int = 2**31 - 1
# #     pd.set_option('display.max_colwidth', None)
# #     pd.set_option('display.max_rows', None)
# #     pd.set_option('display.max_columns', None)
    
# #     if hasattr(pd, 'option_context'):
# #         with pd.option_context('mode.chained_assignment', None):
# #             try:
# #                 # Randomly select a CSV file from the list
# #                 selected_file = random.choice(csv_files)

# #                 # Set the field size limit to handle larger fields
# #                 old_limit = pd.options.display.max_colwidth
# #                 pd.set_option('display.max_colwidth', max_int)
                
# #                 try:
# #                     # Read the CSV file into a DataFrame without headers and specifying that there are no delimiters initially
# #                     df = pd.read_csv(selected_file, header=None, delimiter=None, engine='python')
                    
# #                     # Check if the DataFrame is empty
# #                     if df.empty:
# #                         raise ValueError(f"The file {selected_file} is empty.")
                    
# #                     # Randomly select a row from the DataFrame
# #                     random_row_index = random.randint(0, len(df) - 1)
# #                     selected_row = df.iloc[random_row_index].tolist()  # Convert to list instead of dict since there are no headers
                    
# #                     return selected_row
                
# #                 except Exception as e:
# #                     print(f"Error reading file {selected_file}: {e}")
                
# #                 finally:
# #                     pd.set_option('display.max_colwidth', old_limit)
            
# #             except Exception as e:
# #                 print(f"Exception occurred: {e}")

# def get_random_row_from_csv(csv_files):
#     # Randomly select a CSV file from the list
#     selected_file = random.choice(csv_files)
#     max_int = 2**31 - 1

#     try:
#         # Read the CSV file into a DataFrame without headers and specifying that there are no delimiters initially
#         df = pd.read_csv(selected_file, header=None, delimiter=None, engine='python')
        
#         # Check if the DataFrame is empty
#         if df.empty:
#             raise ValueError(f"The file {selected_file} is empty.")
        
#         # Randomly select a row from the DataFrame
#         random_row_index = random.randint(0, len(df) - 1)
#         selected_row = df.iloc[random_row_index].tolist()  # Convert to list instead of dict since there are no headers
        
#         return selected_row
    
#     except Exception as e:
#         print(f"Error reading file {selected_file}: {e}")
#         return None

# def map_files_to_paths(dataset_dir, csv_list):
#     return [os.path.join(dataset_dir, file) for file in csv_list]

# # CSV Dataset Fetching
# DATASET_DIR = "datasets"
# csv_list = map_files_to_paths(DATASET_DIR, os.listdir(DATASET_DIR))
# row = get_random_row_from_csv(csv_list)

# print(row)

import os
import csv
import random

def get_random_row_from_csv(csv_files):
    # Randomly select a CSV file from the list
    selected_file = random.choice(csv_files)
    
    try:
        # Increase the field size limit
        max_int = 2**31 - 1
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            csv.field_size_limit(int(max_int / 10))
        
        # Open the CSV file
        with open(selected_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)  # Convert to a list of rows
            
            # Check if the file is empty
            if not rows:
                raise ValueError(f"The file {selected_file} is empty.")
            
            # Randomly select a row
            random_row_index = random.randint(0, len(rows) - 1)
            selected_row = rows[random_row_index]
            
            return selected_row  # Return the selected row
        
    except Exception as e:
        print(f"Error reading file {selected_file}: {e}")
        return None

def map_files_to_paths(dataset_dir, csv_list):
    return [os.path.join(dataset_dir, file) for file in csv_list]

# CSV Dataset Fetching
DATASET_DIR = "datasets"
csv_list = map_files_to_paths(DATASET_DIR, os.listdir(DATASET_DIR))
row = get_random_row_from_csv(csv_list)

print(row)