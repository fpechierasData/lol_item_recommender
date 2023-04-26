import json
import os
import requests
import tarfile

def download_json(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded and saved new JSON file at {file_path}")

def download_tarball(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def extract_tarball(file_path, version):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=f"data/{version}")

def get_datadragon_version(local_file_path):
    # Set URL and local JSON file paths
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    local_file_path = "data/version.json"

    # Download JSON if local file doesn't exist
    if not os.path.exists(local_file_path):
        download_json(url, local_file_path)
    else:
        # Load local and remote JSON files
        with open(local_file_path, 'r') as local_file:
            local_data = json.load(local_file)

        remote_data = requests.get(url).json()

        # Compare local and remote JSON files
        if local_data != remote_data:
            download_json(url, local_file_path)
        else:
            print("Local JSON file is already up-to-date.")

def get_datadragon(local_file_path):
    #get version
    with open(local_file_path, 'r') as local_file:
        data = json.load(local_file)
    version = data[0]
    
    #check to see if latest already downloaded
    if os.path.exists(f"data/{version}"):
        print(f"Latest Data Dragon folder present. If you believe this is an error, delete the folder at data/{version} in this project directory and re-run this script.")
        return 0

    #set url, tar path
    url = f"https://ddragon.leagueoflegends.com/cdn/dragontail-{version}.tgz"
    tar_path = f"data/{version}.tgz"

    #download tarball
    print(f"Downloading datadragon version {version}")
    download_tarball(url=url, file_path=tar_path)

    print(f"Unpacking tarball...")
    extract_tarball(file_path=tar_path, version=version)

    print("Deleting tarball...")
    os.remove(tar_path)
    print("Great Success!")

if __name__ == '__main__':
    #check for data folder, make if not present
    if not os.path.exists("data"):
        os.mkdir("data")

    #declare path to write json to, download datadragon
    local_file_path = "data/version.json"
    get_datadragon_version(local_file_path)
    get_datadragon(local_file_path)


