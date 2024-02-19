import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os


# Load required documents
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading{file}...")
        loader = PyPDFLoader(file)

    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading{file}...")
        loader = Docx2txtLoader(file)

    elif extension == ".txt":
        from langchain.document_loaders import TextLoader
        print(f"Loading{file}...")
        loader = TextLoader(file, autodetect_encoding=True)

    else:
        print("Document format not supported!")
        return None

    data = loader.load()
    return data


# Split the docs into chunks for embedding
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# Embed the chunks into vectorstore
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


# Ask questions and get results
def ask_and_get_answers(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever(search_type="similarity",
                                          search_kwargs={"k": k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                        retriever=retriever)
    answer = chain.invoke(q)
    return answer


# Calculate embeddings cost
def calculate_embeddings_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004








