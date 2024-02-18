from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import os
def main():
    if not os.path.exists("chroma_db"):
        loader = DirectoryLoader("./data/python_docs/", glob="**/*.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        documents = text_splitter.split_documents(docs)
        Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="./chroma_db")

if __name__ == "__main__":
    main()