import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv


##load envirnoment variables

load_dotenv()


os.getenv("GROQ_API_KEY")


##get pdf flle

def get_pdf_flle(pdf_docs):
    text = ""
    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text  +=  page.extract_text()
        return text
    

##pdf file word to split into small chunks

def get_text_chunks(text):
    """Splits extracted text into manageable chunks."""
    text_splitter =RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

##create vector store

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)

    vector_store.save("faiss_vector_store")


def get_convesation_chain():
     """Sets up a conversational chain using Groq LLM."""
     prompt_template = """
            Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context."
     Context:
    {context}?

    Question:
    {question}

    Answer:
      
    """
     
     model = ChatGroq(
         temperature=0.3,
         model_name="deepseek-r1-distill-llama-70b",
         groq_api_key=os.getenv("GROQ_API_KEY")

     )


     return load_qa_chain(model, chain_type="stuff")  

def user_input(user_question):
     """Handles user queries by retrieving answers from the vector store."""
     embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

     new_db =  FAISS.load_local("faiss_vector_store" ,embeddings, allow_dangerous_deserialization=True)

     docs = new_db.similarity_search(user_question)

     chain = get_convesation_chain()


     response = chain(
         {"input_documents":docs , "question":user_question},
         return_only_outputs=True,
     )

     st.markdown(f"### Reply:\n{response['output_text']}")





def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":books:", layout="wide")
    st.title("chat with PDF using Groq")

    st.sidebar.header("upload pdf files")
    st.sidebar.markdown("Using Groq ")
    
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF files:",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("submit & process"):
            with st.spinner("processing your files..."):
                 raw_text = get_pdf_flle(pdf_docs)
                 text_chunks= get_text_chunks(raw_text)
                 get_vector_store(text_chunks)
                 st.success("PDFs processed and indexed successfully!")

        user_question = st.text_input("Enter your question:", placeholder="What do you want to know?")

        if user_question:
             with st.spinner("Fetching your answer..."):
                 user_input(user_question)


if __name__ == "__main__":
    main()  