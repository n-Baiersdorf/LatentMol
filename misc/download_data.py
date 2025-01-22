import os
import requests
import gzip
import shutil

def download_and_extract_pubchem_compounds(output_dir, start=1, end=500000, step=500000):
    """
    Downloads and extracts PubChem compound files in .sdf.gz format.

    Parameters:
        output_dir (str): Directory to save the extracted .sdf files.
        start (int): Starting compound range (default: 1).
        end (int): Ending compound range (default: 500000).
        step (int): Step size for compound ranges (default: 500000).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    base_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/"

    for i in range(start, end + 1, step):
        # Generate the filename and URL
        filename = f"Compound_{i:09d}_{i + step - 1:09d}.sdf.gz"
        url = base_url + filename
        
        # Download the file
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            gz_path = os.path.join(output_dir, filename)
            with open(gz_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}.")

            # Extract the .sdf.gz file
            print(f"Extracting {filename}...")
            sdf_path = gz_path[:-3]  # Remove .gz extension
            with gzip.open(gz_path, 'rb') as f_in:
                with open(sdf_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to {sdf_path}.")

            # Optionally delete the .gz file to save space
            os.remove(gz_path)
        else:
            print(f"Failed to download {filename}. HTTP status code: {response.status_code}")

# Example usage
if __name__ == "__main__":
    # Define the output directory and range of compounds to download
    download_and_extract_pubchem_compounds(
        output_dir="data_raw/pubchem_compounds",  # Directory to store extracted files
        start=1,                         # Start of compound range
        end=173000000,                     # End of compound range (adjust as needed)
        step=500000                      # Step size for each file batch
    )


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.