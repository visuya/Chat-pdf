__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import os
import tempfile

#from dotenv import load_dotenv
# load_dotenv()

st.title("CHATPDF")
st.write("---")

#openai key input
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password

#Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    tempfilepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#Treat Uploaded PDF
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)


#split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,   
)

text = text_splitter.split_documents(pages)

#Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
)

#Clear cache
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cashe()


#Chroma
db = Chroma.from_documents(text, embeddings)

#handler streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self,container, initial_text=""):
        self.container = container
        self.text= initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.write(self.text)



#User input
st.header("Ask a question")
question = st.text_input("Enter your question here:") 

if st.button("Submit"):
    with st.spinner("Generating answer..."):
        #Retriever
        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(),
            llm=llm
        )

        prompt = hub.pull("rlm/rag-prompt")

        #generate
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)
        generate_llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0, openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        rag_chain = (
            {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        #question
        result = rag_chain.invoke(question)