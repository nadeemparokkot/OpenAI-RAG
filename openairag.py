import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your api key"

# Provide the path of your PDF file
pdf_path = '/pdf-path/pdf.pdf'

# Initialize the PdfReader
pdfreader = PdfReader(pdf_path)

# Read text from the PDF
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Use CharacterTextSplitter to split the text while keeping token size manageable
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Print the number of chunks for verification
print(f"Total chunks: {len(texts)}")

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Create FAISS index
document_search = FAISS.from_texts(texts, embeddings)

# Initialize the QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Define your query
query = "qn"

# Perform similarity search
docs = document_search.similarity_search(query)

# Run the QA chain
result = chain.run(input_documents=docs, question=query)

# Print the result
print(result)
