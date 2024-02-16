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
        loader = TextLoader(file)

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


if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    st.image("img.png")
    st.subheader("Retrieval Augmented Generative QA")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        # If the user provides a key, load it in an environment variable
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])

        # Number input for chunk size
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2048, value=512)

        # Number of most similar chunks used to assemble the final answer
        k = st.number_input("k", min_value=1, max_value=20, value=3)

        # File chunked, embedded, saved in vectorstore when user clicks button
        add_data = st.button("Add data")

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking, and embedding file..."):
                # Copy file from memory to disk locally
                # Read file content in binary
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                # Load document
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Chunk_size: {chunk_size}, Chunks: {len(chunks)}")

                # Display embeddings cost
                tokens, embeddings_cost = calculate_embeddings_cost(chunks)
                st.write(f"Embeddings cost: ${embeddings_cost:.4f}")

                # Embed the chunks
                vector_store = create_embeddings(chunks)

                # Save the vector store in the streamlit session state
                # To be persistent between reruns
                st.session_state.vs = vector_store
                st.success("File uploaded, chunked and embedded successfully!")

    q = st.text_input("Ask a question about the content of your file")
    # If user asks a question and vector store exists in the session state
    # Load the vector store from the session state into a variable
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f"k:{k}")
            answer = ask_and_get_answers(vector_store, q, k)
            st.text_area("LLM Answer", value=answer)




