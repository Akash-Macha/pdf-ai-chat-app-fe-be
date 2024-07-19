from PyPDF2 import PdfReader
import os
import pickle
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def generate_pkl(pdfContents, filename):
    pdfText = ""
    for page in pdfContents.pages:
        pdfText += page.extract_text()

    # Explained the overlap: https://youtu.be/RIWbalZ7sTo?si=ViGRnWbeV7D14-Rq&t=915
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=pdfText)



    # embeddings
    pdf_name = pdf.name[:-4] # get the name of the pdf, with .pdf
    store_name = f"{pdf_name}.pkl"

    if os.path.exists(store_name):
        with open(store_name, 'rb') as f:
            VectorStore = pickle.load(f)
        st.write("Embeddings loaded from the Disk")
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(store_name, 'wb') as f:
            pickle.dump(VectorStore, f)