from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
import os

pdf_loader = PyPDFLoader("Manual.pdf")
pdf = pdf_loader.load()

print(f"Have Loaded {len(pdf)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(pdf)
print(f"Created {len(chunks)} chunks")


#embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
#embeddings = OllamaEmbeddings(model="all-minilm")
api_key = os.environ["GROQ_API_KEY"]

# Local embeddings (no local Ollama daemon needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Hosted Ollama cloud chat model authenticated with Bearer token
llm = ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    temperature=0,
)
vs = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vc_chroma",
)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},  # top 2 chunks
)

query = "How do I configure the ports?"
returned_chunks = retriever.invoke(query)

##for chunk in returned_chunks:
##    print(f"Score: {'{'}chunk.metadata.get('score', 'N/A'){'}'}")
##    print(chunk.page_content[:200])
##    print("---")
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


prompt_template = PromptTemplate.from_template("""
You are a assistant providing direction on maintenance on a WAN switch. Limit your answers to the provided context.
Answer to the best of your ability in the provided domain."

Context:
{context}

Question: {question}

Answer (cite the relevant section if possible):
""")

context_text = "\n\n---\n\n".join([c.page_content for c in returned_chunks])

chain = prompt_template | llm

response = chain.invoke({
    "context": context_text,"question": query,
})

print(response.content)
