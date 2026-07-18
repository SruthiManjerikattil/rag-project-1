from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

chat_model = ChatHuggingFace(llm=llm)

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
    return results
    print("\nTop 3 retrieved chunks:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content)
        print()


def generate_answer(chat_model, question, chunks):
    context = "\n\n".join([doc.page_content for doc in chunks])

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so — don't make things up.

    Context:
        {context}

    Question: {question}

    Answer:"""
    response = chat_model.invoke(prompt)
    return response.content

def main():
    documents = load_documents("data")
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks, embeddings)

    question = input("Enter your question: ")
    chunks_retrieved = retrieve_chunks(vectorstore, question)
    answer = generate_answer(chat_model, question, chunks_retrieved)
    print("\nAnswer:", answer)


if __name__ == "__main__":
    main()