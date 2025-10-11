#load pdf
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(r"C:\Users\hp\Downloads\DSM-5-TR.pdf")
documents = loader.load()
print(f"Total pages loaded: {len(documents)}")

#split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

