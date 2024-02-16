# What is this project about?

This is a question-answering app that allows users to provide a large 
language model(LLM) with custom data the LLM has not been trained on, so
that it can perform Q&A based on that data. 

It uses model inputs and embeddings-based search with the following stack: 
LangChain, OpenAI and Chroma. Streamlit for the front-end.

Pipeline:
1. Prepare search data:
   - load data into LangChain Documents
   - split data into chunks
   - embed chunks into vectors
   - save chunks and embeddings into the vector database
2. Search: 
    - embed the user question
    - rank vectors by similarity to the question's embeddings
3. Ask questions:
    - insert the question and most relevant chunks into a message for the GPT model
    - Return the GPT's answer
   