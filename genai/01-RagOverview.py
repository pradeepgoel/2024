
# %%
import os
import langchain
import openai
import sys
import numpy as np
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ['OPENAI_API_KEY']

# %%
# Load a pdf file using langchain document loaders. Multiple documents can be loaded as well. 

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/Southern Region CP7 Strategic Business Plan.pdf")

pages = loader.load_and_split()

len(pages)

# %%
# Add sample code for loading multiple documents
loaders = [
# add file path here for all files

]

# %%
# Review examine a page that is loaded into pages object
pages[7]

# %%
# Split document using Recursive Character Text Splitter. It splits on /n/n, /n, ".", ",", " ", ""
# Experiment with different chunk size and check resulting splits


from langchain.text_splitter import RecursiveCharacterTextSplitter

r_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 20
)

splits = r_spliter.split_documents(pages)

len(splits)



# %%
# Use Chroma database to store all chuncks and persist them into a directory.
# Use !rm -rf, for removing old database file 

from langchain.vectorstores import Chroma
persist_directory = 'data/mysplits_11Jan'
# !rm -rf ./data/mysplits

# %%
# Use OpenAI embeddings to embed all splits. Create an embeding_model object and pass that to Chroma DB for embedding.
from langchain.embeddings import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

# %%


vectordb = Chroma.from_documents(

    documents= splits,
    embedding=embeddings_model,
    #persist_directory=persist_directory
    
)

# %%
print(vectordb._collection.count())

# %%
vectordb.persist()

# %%
question = "How many freight terminal are there?, Please list of first 5 terminals."

# %%
question1 = "Does this document have information on Covid and its impact on revenue?"

# %%
docs = vectordb.similarity_search(question, k=3)

# %%
docs1 = vectordb.similarity_search(question1, k=5)

# %%
docs1[2]

# %%
docs[0]

# %%
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model= "gpt-3.5-turbo", temperature=0)

# %%
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = vectordb.as_retriever()
)

# %%
result = qa_chain({"query" : question})

result

# %%
result1 = qa_chain({"query" : question1})

result1


# %%
# Prompt

from langchain.prompts import PromptTemplate
# Build prompt

template = """ Use the following pieces of context to answer the question at the end. Add name of the documents at the end. take it from the context data. {context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# %%
qa_chain1 = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# %%
question = "Does this document have any information on COVID impact?"

# %%
result1 = qa_chain1({"query" : question})

# %%
result1["result"]

# %%
result["source_documents"]


