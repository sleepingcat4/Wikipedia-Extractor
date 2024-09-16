import requests
from bs4 import BeautifulSoup
import os
import gzip
import json
import base64
import pyarrow as pa
import pyarrow.parquet as pq
import regex as re
from tqdm import tqdm
import random
import string
import shutil
import time

def extract_links(url, prefix, ends_with=None, return_first=False):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    
    matching_links = []
    for link in links:
        href = link['href']
        if href.startswith(prefix):
            if ends_with and not href.endswith(ends_with):
                continue
            parts = href.split('-')
            if len(parts) > 1 and parts[1].isdigit():
                numerical_value = parts[1]
                full_link = url + href
                matching_links.append((numerical_value, full_link))
                if return_first:
                    return matching_links
    
    return matching_links

def download_file(link, directory='.'):
    response = requests.get(link, stream=True)
    filename = os.path.join(directory, link.split('/')[-1])
    
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
    
    print(f"Downloaded: {filename}")
    return filename

def format_title(title):
    if title:
        return title.replace(' ', '_')
    return None

def clean_text(text):
    if text:
        return re.sub(r'[^\p{L}\s]', '', text, flags=re.UNICODE)
    return None

def validate_output_file(output_file_path):
    if not output_file_path.endswith(".parquet"):
        raise ValueError("Output file must have a .parquet extension.")

def open_file(file_path):
    if file_path.endswith('.json.gz'):
        return gzip.open(file_path, 'rt', encoding='utf-8')
    elif file_path.endswith('.json'):
        return open(file_path, 'r')
    else:
        raise ValueError("Unsupported file format. Only .json and .json.gz are allowed.")

def save_checkpoint(data, checkpoint_num, checkpoint_folder, column_name):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_{checkpoint_num}.parquet")
    table = pa.Table.from_pydict({
        'URL': [d['URL'] for d in data],
        'Wiki': [d['Wiki'] for d in data],
        'Language': [d['Language'] for d in data],
        'Title': [d['Title'] for d in data],
        column_name: [d[column_name] for d in data],
        'Version Control': [d['Version Control'] for d in data],
        'Popularity Score': [d['Popularity Score'] for d in data]
    })
    pq.write_table(table, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def count_rows_in_parquet(file_path):
    # Start timing
    start_time = time.time()

    # Validate file path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Open the Parquet file and count the rows
    parquet_file = pq.ParquetFile(file_path)
    num_rows = parquet_file.metadata.num_rows

    # End timing
    end_time = time.time()

    print(f"Number of rows: {num_rows}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

def extract_language_code(prefix):
    return prefix[:2]  # Assumes language code is the first 2 characters of the prefix

def main():
    url = 'https://dumps.wikimedia.org/other/cirrussearch/current/'
    user_prefix = input("Enter the prefix (e.g., nlwiki or enwiki): ")
    language_code = extract_language_code(user_prefix)
    
    print("The extraction process may take a few hours to a day to complete. Please be patient.")
    
    filter_option = input("Do you want only links that end with 'cirrussearch-content.json.gz'? (yes/no): ")
    ends_with = 'cirrussearch-content.json.gz' if filter_option.lower() == 'yes' else None
    first_only = input("Do you want only the first matching link? (yes/no): ").lower() == 'yes'

    links_with_prefix = extract_links(url, user_prefix, ends_with, return_first=first_only)

    if links_with_prefix:
        for num, link in links_with_prefix:
            print(f"Numerical Value: {num}, Link: {link}")
            downloaded_file = download_file(link)
            base_value = num  # Store the base value for later use
            break
    else:
        print(f"No links found with prefix '{user_prefix}' and ending with '{ends_with}'" if ends_with else f"No links found with prefix '{user_prefix}'")
        return

    extract_option = input("Do you want to extract abstract, full text, or both? (abstract/full_text/both): ").lower()
    clean_text_flag = input("Do you want to enable text cleaning? (yes/no): ").strip().lower()
    checkpoint_interval = int(input("Enter the checkpoint interval (number of rows per checkpoint): ").strip())
    checkpoint_folder = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

    # Output file names
    abstract_output_file = f"{user_prefix}_abstract.parquet"
    full_text_output_file = f"{user_prefix}_full_text.parquet"

    data_abstract = []
    data_full_text = []
    checkpoint_data_abstract = []
    checkpoint_data_full_text = []
    extract_all = True
    limit = 50
    processed_count = 0
    checkpoint_num = 0

    with open_file(downloaded_file) as file:
        for i, line in enumerate(file):
            if not extract_all and processed_count >= limit:
                break
            
            entry = json.loads(line.strip())
            
            wiki = entry.get('wiki', None)
            language = entry.get('language', None)
            title = entry.get('title', None)
            full_text = entry.get('full_text', None) if extract_option in ['full_text', 'both'] else None
            abstract = entry.get('opening_text', None) if extract_option in ['abstract', 'both'] else None
            popularity_score = entry.get('popularity_score', None)
            
            if all([wiki, language, title, (full_text if full_text else abstract)]):
                formatted_title = format_title(title)
                url = f"https://{language_code}.wikipedia.org/wiki/{formatted_title}" if formatted_title else None
                
                if clean_text_flag == 'yes':
                    if full_text:
                        full_text = clean_text(full_text)
                    if abstract:
                        abstract = clean_text(abstract)
                
                version_control_value = base_value + str(processed_count + 1)
                version_control_bytes = version_control_value.encode('utf-8')
                version_control_base64 = base64.b64encode(version_control_bytes).decode('utf-8')
                
                entry_data = {
                    'URL': url,
                    'Wiki': wiki,
                    'Language': language,
                    'Title': title,
                    'Version Control': version_control_base64,
                    'Popularity Score': popularity_score
                }

                if extract_option in ['abstract', 'both']:
                    entry_data['Abstract'] = abstract
                    data_abstract.append(entry_data)
                    checkpoint_data_abstract.append(entry_data)
                    
                if extract_option in ['full_text', 'both']:
                    entry_data['Full Text'] = full_text
                    data_full_text.append(entry_data)
                    checkpoint_data_full_text.append(entry_data)
                
                processed_count += 1
                
                if processed_count % checkpoint_interval == 0:
                    checkpoint_num += 1
                    if checkpoint_data_abstract:
                        save_checkpoint(checkpoint_data_abstract, checkpoint_num, checkpoint_folder, 'Abstract')
                        checkpoint_data_abstract = []
                    if checkpoint_data_full_text:
                        save_checkpoint(checkpoint_data_full_text, checkpoint_num, checkpoint_folder, 'Full Text')
                        checkpoint_data_full_text = []
                
                print(f"Processed entry {processed_count}")

    if checkpoint_data_abstract:
        checkpoint_num += 1
        save_checkpoint(checkpoint_data_abstract, checkpoint_num, checkpoint_folder, 'Abstract')

    if checkpoint_data_full_text:
        checkpoint_num += 1
        save_checkpoint(checkpoint_data_full_text, checkpoint_num, checkpoint_folder, 'Full Text')

    # Write data to separate Parquet files
    if data_abstract:
        table_abstract = pa.Table.from_pydict({
            'URL': [d['URL'] for d in data_abstract],
            'Wiki': [d['Wiki'] for d in data_abstract],
            'Language': [d['Language'] for d in data_abstract],
            'Title': [d['Title'] for d in data_abstract],
            'Abstract': [d['Abstract'] for d in data_abstract],
            'Version Control': [d['Version Control'] for d in data_abstract],
            'Popularity Score': [d['Popularity Score'] for d in data_abstract]
        })
        pq.write_table(table_abstract, abstract_output_file)
        print(f"Data with abstracts saved to {abstract_output_file}")

    if data_full_text:
        table_full_text = pa.Table.from_pydict({
            'URL': [d['URL'] for d in data_full_text],
            'Wiki': [d['Wiki'] for d in data_full_text],
            'Language': [d['Language'] for d in data_full_text],
            'Title': [d['Title'] for d in data_full_text],
            'Full Text': [d['Full Text'] for d in data_full_text],
            'Version Control': [d['Version Control'] for d in data_full_text],
            'Popularity Score': [d['Popularity Score'] for d in data_full_text]
        })
        pq.write_table(table_full_text, full_text_output_file)
        print(f"Data with full texts saved to {full_text_output_file}")

    shutil.rmtree(checkpoint_folder, ignore_errors=True)
    print(f"Checkpoint folder '{checkpoint_folder}' deleted.")

if __name__ == "__main__":
    main()
