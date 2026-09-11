import os
import time
import arxiv
import gradio as gr
import shutil
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import RootModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
# Search arXiv for papers and download them
def process_papers(query, question_text):
    dirpath = "arxiv_papers"
    qdrant_path = "./tmp/local_qdrant"

    # Clean old data
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)

# Search arXiv for papers related to "LLM"
    client = arxiv.Client()
    search = arxiv.Search(
        query = query,
        max_results=10,
        sort_order=arxiv.SortOrder.Descending
)

    # Download and save the papers
    for result in client.results(search):
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                print(f"--> Paper id {result.get_short_id()} with title {result.title} is downloaded.")
                break
            except (FileNotFoundError,ConnectionResetError) as e:
                print("Error occured:",e)
                time.sleep(5)

    # Load papers, concatenate them, andd split into chunks
    papers = []
    loader = DirectoryLoader(dirpath, glob="./*.pdf",loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        print(f" Error laoding file :{e}")
    print("Total number of pages loaded:", len(papers))

    # Concatenate all pages content into a single string
    full_text = ''
    for paper in papers:
        full_text += paper.page_content

    # Remove empty lines and join lines into a single string
    full_text = " ".join(line for line in full_text.splitlines() if line)
    print("Total characters in the concatenated text:",len(full_text))

    #Split the text into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])

    # Create Qdrant vector store and store embeddings
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path="./tmp/local_qdrant",
        collection_name = "arxiv_papers",
    )
    retriever = qdrant.as_retriever()

    # 4. Define prompt template and initialize Ollama
    template = """Answer the question briefly based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize Ollama
    ollama_llm = "tinyllama"
    model = ChatOllama(model=ollama_llm)

    # Define the processing chain
    chain = (
        RunnableParallel({"context": retriever,"question": RunnablePassthrough()})
        | prompt 
        | model
        | StrOutputParser()
    )

    # Add typing for input
    class Question(RootModel[str]):
        pass

    chain = chain.with_types(input_type = Question)

    # Ask a question

    result = chain.invoke(question_text)
    return result

iface = gr.Interface(
    fn = process_papers,
    inputs = ['text','text'],
    outputs = 'text',
    description = 'Enter a search query and a question to process arXiv papers.'
)

iface.launch()