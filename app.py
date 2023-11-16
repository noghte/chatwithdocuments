from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
import os
import openai

load_dotenv()
# openai.api_key = os.environ.get("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_KEY")

def split_documents(documents, chunk_size=100, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)
    return docs

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    # result = llm.predict("Hello! What is your name?")
    # print(result)
    loader = DirectoryLoader("./texts")
    documents = loader.load()
    print("Documents:", len(documents))
    docs = split_documents(documents)
    print("Splitted documents:", len(docs))
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma.from_documents(docs, embeddings)
    
    chain = load_qa_chain(llm, chain_type="stuff")

    query = "What should I do if I forget my GSU password?"
    matching_documents = db.similarity_search(query)
    
    answer = chain.run(input_documents=matching_documents, question=query)
    print(answer)
    

    