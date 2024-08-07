"""Create a ChromaDB given a set of input text files.
 Multiple files
 Multiple file formats (PDF, TXT, CSV)
 ChromaDB
"""

import os
import sys
import zipfile
import tempfile
import urllib.request
from tqdm import tqdm

from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import MergedDataLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import load_config, set_progress

# add the directory for locally hosted models
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/huggingface_cache"


def check_input(input_path):
    """Check type of input data.

    Verify that a given input data is a path to a directory, zip file, or URL.

    Args:
        input_path: Name of file, directory, or URL (string).

    Returns:
        Type of input (string).
    """
    if os.path.isdir(input_path):
        return "directory"

    # Check if input is a zip file
    if os.path.isfile(input_path) and input_path.endswith(".zip"):
        return "zip"

    # Check if input is a URL
    if input_path.startswith("http://") or input_path.startswith("https://"):
        return "url"

    return "other"


def unzip_file(zip_file, extract_dir):
    """Unzip file.

    Args:
        zip_file: File to unzip (string).
        extract_dir: Destination of unzipped data (string).
    """
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def download_and_unzip(url, tmp_dir):
    """Download file

    Args:
        url: URL of file to download (string).
        tmp_dir: Temporary destination folder of downloaded data (string).
    """
    file_name = os.path.join(tmp_dir, os.path.basename(url))
    urllib.request.urlretrieve(url, file_name)
    # Check if downloaded file is a zip file
    if file_name.endswith(".zip"):
        unzip_file(file_name, tmp_dir)


def load_data(directory):
    """Load data.

    Find and load all txt, PDF, and CSV files.

    Args:
        dir: Directory name (string).

    Raises:
        ValueError: if no files were found.
    """
    loader_txt = DirectoryLoader(directory, glob="./**/*.txt", loader_cls=TextLoader)
    loader_pdf = DirectoryLoader(directory, glob="./**/*.pdf", loader_cls=PyPDFLoader)
    loader_csv = DirectoryLoader(directory, glob="./**/*.csv", loader_cls=CSVLoader)
    loader = MergedDataLoader(loaders=[loader_txt, loader_pdf, loader_csv])

    documents = loader.load()
    if not len(documents) > 0:
        raise ValueError(f"Error: No .txt of .pdf files in '{directory}'.")

    return documents


def prep_data(in_data):
    """
    Fetch and load data.

    Parameters:
        data: Input data as a directory, zip file, or URL pointing to a zip file
              (string).

    Raises:
        ValueError: If data is not a valid type.
    """
    tmp_dir = tempfile.mkdtemp()
    input_type = check_input(in_data)
    if input_type == "directory":
        documents = load_data(in_data)
    elif input_type == "zip":
        unzip_file(in_data, tmp_dir)
        documents = load_data(tmp_dir)
    elif input_type == "url":
        download_and_unzip(in_data, tmp_dir)
        documents = load_data(tmp_dir)
    else:
        raise ValueError(
            f"Error: Data '{in_data}' must be a directory, a" "zip file, or a URL."
        )

    return documents


def create_db(data, db_dir, config_file):
    """Create a Chroma database.

    Args:
        data: Path to data to be transformed and stored as a Chroma database.
        db_dir: Path to directory that will contain the database.
        conf: Path to the file containing the parameter values.

    Raises:
        ValueError: If target directory for database already exists.
    """

    prog = tqdm(range(100), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    set_progress(prog, 2, "Loading configuration data")
    config = load_config(config_file)

    set_progress(prog, 1, "Checking target DB '{db_dir}' doesn't exist")
    if os.path.exists(db_dir):
        raise ValueError(
            f"Error: Directory '{db_dir}' already exists. Please choose a"
            "different name."
        )

    set_progress(prog, 2, "Fetching and loading documents")
    documents = prep_data(data)

    set_progress(prog, 10, "Transforming and splitting data")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
    )
    texts = text_splitter.split_documents(documents)

    set_progress(prog, 30, "Processing data")
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
    )

    set_progress(prog, 15, "Creating database")
    Chroma.from_documents(
        documents=texts, embedding=instructor_embeddings, persist_directory=db_dir
    )
    set_progress(prog, 40, "Database created successfully")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Error: wrong number of arguments")
        print("Usage: python create_db.py <data> <db_directory> [config file]")
        sys.exit(1)

    dat = sys.argv[1]
    db = sys.argv[2]
    CONF = "./scripts/config.yaml"
    if len(sys.argv) == 4:
        CONF = sys.argv[3]
    create_db(dat, db, CONF)
