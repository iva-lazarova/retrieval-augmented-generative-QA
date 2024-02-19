import streamlit as st
import os
from data_opps import (load_document, chunk_data, create_embeddings,
                     ask_and_get_answers, calculate_embeddings_cost)


st.image("img.png")
st.subheader("Retrieval Augmented Generative QA")


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]


with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    # If the user provides a key, load it in an environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])

    # Number input for chunk size
    chunk_size = st.number_input("Chunk Size",
                                 min_value=100,
                                 max_value=2048,
                                 value=512,
                                 on_change=clear_history)

    # Number of most similar chunks to be assembled for final answer
    k = st.number_input("k",
                        min_value=1,
                        max_value=20,
                        value=3,
                        on_change=clear_history)

    # File chunked, embedded, saved in vectorstore when button clicked
    add_data = st.button("Add data", on_click=clear_history)

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
        # st.write(f"k:{k}")
        answer = ask_and_get_answers(vector_store, q, k)
        st.text_area("LLM Answer:", value=answer)

        st.divider()
        # If the history key not in session state, create it
        if "history" not in st.session_state:
            st.session_state.history = ""

        # Concat current question and its answer
        value = f'Q: {q} \nA: {answer}'
        # Display latest question and answer before history
        st.session_state.history = f"{value} \n {'-' * 100} \n {st.session_state.history}"

        # Take chat history from session state into a variable
        # Display variable in main text area
        chat = st.session_state.history
        st.text_area(label="Chat History", value=chat, key="history", height=400)

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print("Hi!")
