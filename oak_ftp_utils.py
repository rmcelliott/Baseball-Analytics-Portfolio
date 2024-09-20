"""
    Project Title: Ballers Baseball FTP Access/Download Tool

    Authors: Ben Jones and Riley Elliott

    Date:   9/19/2024
"""

import argparse
import re
import os
import pandas as pd
from ftplib import FTP
from io import BytesIO


# Regex pattern to extract verified venue name from 2024 filename
VENUE_REGEX = r'^2024\d{4}-(.*?)-\d\.csv'

# # Regex pattern to extract unverified venue name from 2024 filename
# VENUE_REGEX = r'^2024\d{4}-(.*?)-\d_unverified\.csv'


# Function to list files in directories (FTP)
def list_ftp_files(ftp, path, file_type):
    files = []
    try:
        ftp.cwd(path)
    except Exception as e:
        print(f"Error changing directory to {path}: {e}")
        return files

    items = ftp.nlst()
    for item in items:
        item_path = f"{path}/{item}"
        try:
            print(f'Looking in {item_path}')
            ftp.cwd(item_path)
            files.extend(list_ftp_files(ftp, item_path, file_type))
        except Exception as e:
            if file_type == "Verified":
                if item.lower().endswith('.csv') and item.startswith('2024'):
                    files.append(item_path)
                    print(f'File found at {item_path}')
                else:
                    print(f'Error accessing {item_path}: {e}')
            elif file_type == "Unverified":
                if item.lower().endswith('unverified.csv') and item.startswith('2024'):
                    files.append(item_path)
                    print(f'File found at {item_path}')
                else:
                    print(f'Error accessing {item_path}: {e}')
    return files


# Function to list files in directories (Local)
def list_local_files(path, file_type):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if file_type == "Verified":
                if filename.lower().endswith('.csv') and filename.startswith('2024'):
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
                    print(f'File found at {file_path}')
            elif file_type == "Unverified":
                if filename.lower().endswith('unverified.csv') and filename.startswith('2024'):
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
                    print(f'File found at {file_path}')

    return files


# Function to download and read CSV file from FTP
def download_csv_ftp(ftp, file_path):
    with BytesIO() as bio:
        ftp.retrbinary(f"RETR {file_path}", bio.write)
        bio.seek(0)
        return pd.read_csv(bio)


# Function to read CSV file from local path
def download_csv_local(file_path):
    return pd.read_csv(file_path)


# Function to concatenate CSV files based on a list of venues
def concatenate_venue_files(files, venues, download_func, *args):
    df_list = []
    for file in files:
        match = re.match(VENUE_REGEX, os.path.basename(file))
        if match and match.group(1) in venues:
            print(f'Downloading file: {file}')
            df = download_func(*args, file) if args else download_func(file)
            df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


# Main function to handle the argument parsing and logic
def main():

    # Specify desired file type ("Verified" or "Unverified")
    file_type = "Verified"

    parser = argparse.ArgumentParser(description="Concatenate CSV files from an FTP server or local directory based on venue name.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", action="store_true", help="Concatenate all files with a valid venue name.")
    group.add_argument("-pl", action="store_true", help="Concatenate files with venues in the specified list.")
    group.add_argument("-oak", action="store_true", help="Concatenate files with the venue 'PomonaPitzer'.")
    parser.add_argument("-debug", action="store_true", help="Use local directory for debug.")

    args = parser.parse_args()

    if args.debug:
        # Local debug mode
        local_base_path = './local_ftp'
        print(f"Debug mode: Using local directory {local_base_path}")

        # List all CSV files in the local directory
        print('Starting file discovery in local directory...')
        files = list_local_files(local_base_path, file_type)
        print('File discovery complete.')

        # Determine the list of venues to filter by
        if args.a:
            venues = list(set(re.match(VENUE_REGEX, os.path.basename(file)).group(1) for file in files if re.match(VENUE_REGEX, os.path.basename(file))))
            filename = './output/a.csv'
        elif args.pl:
            venues = ['CenteneStadium', 'FlatheadField', 'OgrenPark', 'MemorialStadium', 'DehlerPark', 'MelaleucaField']
            filename = './output/pl.csv'
        elif args.oak:
            venues = ['RaimondiPark']
            filename = './output/oak.csv'

        print(f'Venues to filter by: {venues}')

        # Concatenate files based on the specified venues
        concatenated_df = concatenate_venue_files(files, venues, download_csv_local)

    else:
        # FTP mode
        ftp_host = "ftp.trackmanbaseball.com"
        # ftp_user = "Pomona"
        # ftp_pass = "2e=rCAyf&v"
        ftp_user = "Oakland Ballers"
        ftp_pass = "LHacXq7Q53"

        # Connect to FTP server
        ftp = FTP(ftp_host)
        ftp.login(user=ftp_user, passwd=ftp_pass)

        # Set the base path to the current working directory
        ftp_base_path = ftp.pwd()
        print(f"Set base path to current working directory: {ftp_base_path}")

        # List contents of the current working directory
        try:
            items = ftp.nlst()
            print(f"Contents of the current directory: {items}")
        except Exception as e:
            print(f"Error listing contents of the current directory: {e}")
            ftp.quit()
            return

        # Verify base path
        try:
            ftp.cwd(ftp_base_path)
            print(f"Changed directory to base path: {ftp_base_path}")
        except Exception as e:
            print(f"Error changing directory to base path {ftp_base_path}: {e}")
            ftp.quit()
            return

        # List all CSV files on the server
        print('Starting file discovery...')
        files = list_ftp_files(ftp, ftp_base_path, file_type)
        print('File discovery complete.')

        # Determine the list of venues to filter by
        if args.a:
            venues = list(set(re.match(VENUE_REGEX, os.path.basename(file)).group(1) for file in files if re.match(VENUE_REGEX, os.path.basename(file))))
            filename = 'all.csv'
        elif args.pl:
            venues = ['CenteneStadium', 'FlatheadField', 'OgrenPark', 'MemorialStadium', 'DehlerPark', 'MelaleucaField']
            filename = 'pl.csv'
        elif args.oak:
            venues = ['RaimondiPark']
            filename = 'oak.csv'

        print(f'Venues to filter by: {venues}')

        # Concatenate files based on the specified venues
        concatenated_df = concatenate_venue_files(files, venues, download_csv_ftp, ftp)

        # Close the FTP connection
        ftp.quit()

    # Save the concatenated DataFrame to a CSV file
    concatenated_df.to_csv(filename, index=False)
    print(f'Saved concatenated CSV as {filename}')


if __name__ == "__main__":
    main()
