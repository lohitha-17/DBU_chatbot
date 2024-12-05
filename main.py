import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import textwrap  # For wrapping long text

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-key"
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def create_embedding_store(chunks):
    embeddings = OpenAIEmbeddings()
    metadata = [{"source": f"Chunk {i}"} for i in range(len(chunks))]
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadata)
    return vector_store

def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

def format_response(response, width=150):
    """Format the response to wrap text neatly."""
    return "\n".join(textwrap.wrap(response, width))

if __name__ == "__main__":
    pdf_path = "cal_catalog.pdf"
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found!")
        exit(1)

    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(pdf_text)

    print("Creating embeddings...")
    vector_store = create_embedding_store(chunks)

    print("Setting up the query system...")
    qa_chain = build_qa_chain(vector_store)

    print("\nPDF is ready for querying! Ask me anything.")
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye! Have a great day!")
            break
        elif query.lower() in ['hi', 'hello', 'hey']:
            print("Hi there! How can I assist you today?")
            continue
        elif query.lower() in ['how are you?', 'how’s it going?']:
            print("I'm just a program, but I'm here and ready to help! How can I assist you?")
            continue

        try:
            response = qa_chain.invoke({"question": query})
            source_documents = response.get("source_documents", [])
            answer = response.get("result") or response.get("answer") or response.get("output_text", "Sorry, I couldn't find an answer.")
            if answer == "Sorry, I couldn't find an answer.":
                print("\nI'm not sure about that. Maybe try rephrasing your question?")
            else:
                print(f"\n{format_response(answer)}")
        except Exception as e:
            print(f"An error occurred: {e}")