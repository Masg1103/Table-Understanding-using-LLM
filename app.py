import tempfile
import camelot
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

def extract_tables_from_pdf(uploaded_pdf):
    # Save the PDF from the upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(uploaded_pdf.getbuffer())
        temp_pdf_path = temp_pdf.name

    # Now that we have a file path, we can use Camelot to read the PDF
    tables = camelot.read_pdf(temp_pdf_path, pages='all', flavor='lattice')

    return [table.df for table in tables]

def preprocess_table_data(dataframes):
    # This is an example function that you will need to adapt to your specific data
    # For simplicity, let's just concatenate all tables into one big table
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Perform any additional preprocessing here...
    return combined_df


def answer_question(question, data):
    # Split data into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(data)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create a knowledge base
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Search the knowledge base for documents related to the user's question
    docs = knowledge_base.similarity_search(question)

    # Initialize an OpenAI model
    llm = OpenAI()

    # Load a question-answering chain using the OpenAI model
    chain = load_qa_chain(llm, chain_type="stuff")

    # Generate a response
    response = chain.run(input_documents=docs, question=question)
    return response

def main():
    st.title('PDF Table Extractor and Question Answering System')

    uploaded_pdf = st.file_uploader("Upload a PDF containing tables", type="pdf")
    
    if uploaded_pdf:
        # Extract tables from the PDF
        try:
            tables = extract_tables_from_pdf(uploaded_pdf)
            for table_df in tables:
                table_df
                 
        
            st.success('Tables extracted successfully!')
        except Exception as e:
            st.error(f'An error occurred when extracting tables: {e}')
            return

        # Preprocess the data
        preprocessed_data = preprocess_table_data(tables)

        # Displaying the extracted table for user reference
        st.subheader('Extracted Table Data')
        
        # Convert the preprocessed data to a string for GPT-3
        data_as_string = preprocessed_data.to_string(index=False)
        
        # User inputs a question
        user_question = st.text_input("Enter your question related to the table(s)")
        
        if user_question:
            # Utilize Language Model
            answer = answer_question(user_question, data_as_string)

            # The answer
            st.subheader('Answer')
            st.write(answer)

if __name__ == "__main__":
    main()
