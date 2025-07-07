import zipfile


def unzip_file(zip_file_path, extract_to_path):
    """
    Unzips a zip file to a specified directory.

    Args:
        zip_file_path (str): The path to the zip file.
        extract_to_path (str): The directory where the contents will be extracted.
    """
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Successfully unzipped {zip_file_path} to {extract_to_path}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_file_path} is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: Zip file not found at {zip_file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
