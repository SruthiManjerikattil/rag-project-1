from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_documents(data_folder="data"):
    documents = []
    data_path = Path(data_folder)

    for file_path in data_path.iterdir():
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks, embeddings):
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def retrieve_chunks(vectorstore, question):
    results = vectorstore.similarity_search(question, k=3)

    print("\nTop 3 retrieved chunks:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content)
        print()

def main():
    documents = load_documents("data")
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks, embeddings)

    question = input("Enter your question: ")
    retrieve_chunks(vectorstore, question)


if __name__ == "__main__":
    main()