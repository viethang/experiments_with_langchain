from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
URL = "http://0.0.0.0:8000/v1"
KEY = "tata"
llm = ChatOpenAI(openai_api_key=KEY, base_url=URL,
                 model="mistralai/Mistral-7B-Instruct-v0.2")

loader = WebBaseLoader("https://www.marmiton.org/recettes/recette_tarte-aux-pommes-de-terre-et-au-roquefort_26889.aspx")

docs = loader.load()

embeddings = HuggingFaceInstructEmbeddings(
)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
input = "Extract the list of ingredients and preparation step for a dish from the context. Display the results in English then translate it into French."
response = retrieval_chain.invoke(
    {"input": input })
print(response["answer"])
