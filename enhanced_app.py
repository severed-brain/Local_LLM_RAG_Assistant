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

def process_papers(query, question_text, num_papers, progress=gr.Progress(track_tqdm=True)):
    dirpath = "arxiv_papers"
    qdrant_path = "./tmp/local_qdrant"

    # Clean old data
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)

    # Search arXiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=int(num_papers),
        sort_order=arxiv.SortOrder.Descending
    )

    for i, result in enumerate(client.results(search), 1):
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                print(f"--> Paper {i}/{num_papers}: {result.title} downloaded.")
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error:", e)
                time.sleep(5)

    # Load papers
    loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        return f"Error loading papers: {e}", None

    total_pages = len(papers)

    # Concatenate text
    full_text = " ".join(line for doc in papers for line in doc.page_content.splitlines() if line)
    total_chars = len(full_text)

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])

    # Qdrant vector store
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path=qdrant_path,
        collection_name="arxiv_papers",
    )
    retriever = qdrant.as_retriever()

    # Descriptive prompt
    template = """You are an expert research assistant. Based only on the following context from research papers, provide a detailed and well-structured answer to the question. Cite relevant insights clearly. You can provide your knowledge which is out of the context when necessary while framing the answer. The final answer should be containing minimum 200 words.

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOllama(model="tinyllama")

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    class Question(RootModel[str]):
        pass

    chain = chain.with_types(input_type=Question)
    result = chain.invoke(question_text)

        # Stats calculations
    word_count = len(full_text.split())
    avg_chars_per_page = total_chars / total_pages if total_pages else 0
    num_chunks = len(paper_chunks)
    avg_chunk_length = total_chars / num_chunks if num_chunks else 0

    stats_table = [
        ["PDFs Downloaded", int(num_papers)],
        ["Pages Extracted", total_pages],
        ["Characters Extracted", total_chars],
        ["Words Extracted", word_count],
        ["Average Characters per Page", round(avg_chars_per_page, 2)],
        ["Number of Chunks Created", num_chunks],
        ["Average Chunk Length", round(avg_chunk_length, 2)],
        ["Embedding Model Used", "GPT4AllEmbeddings"],
        ["LLM Used", "tinyllama"],
        ["Search Query", query]
    ]

    return result, stats_table

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Research Assistant with arXiv + Qdrant + Ollama")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Search Query (e.g., transformers)")
            num_papers = gr.Number(label="Number of PDFs to Download", precision=0, value=5)
            question_input = gr.Textbox(label="Your Question", lines=2)
            run_btn = gr.Button("🔍 Search, Extract & Answer")
            answer_output = gr.Textbox(label="Answer", lines=6)
        with gr.Column(scale=1):
            stats_output = gr.Dataframe(headers=["Metric", "Value"], label="Extraction Stats", interactive=False)

    run_btn.click(
        fn=process_papers,
        inputs=[query_input, question_input, num_papers],
        outputs=[answer_output, stats_output]
    )

demo.launch()
