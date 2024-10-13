import os
import sys
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredOrgModeLoader,
)

def load_text(file_path):
    print(f"Loading {file_path}")
    print()
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs

def load_org(file_path):
    print(f"Loading {file_path}")
    print()
    loader = UnstructuredOrgModeLoader(file_path)
    docs = loader.load()
    return docs

def load_text_in_dir(directory):
    loader = DirectoryLoader(directory, glob="**/*.txt", recursive = True, loader_cls=TextLoader, silent_errors=True)
    docs = loader.load()
    print(f"Totally {len(docs)} files are loaded.")
    print()
    return docs

def load_org_in_dir(directory):
    org_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".org"):
                org_files.append(os.path.join(root, file))
    print(f"Totally {len(org_files)} files are loaded.")
    print()
    all_docs = []
    for file_path in org_files:
        all_docs.extend(load_org(file_path))
    return all_docs
