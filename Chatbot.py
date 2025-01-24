import os
import glob
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)

def load_environment_variables():
    # Lade den OpenAI API-Schlüssel aus den Umgebungsvariablen
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY Umgebungsvariable nicht gesetzt")
    return openai_api_key

def load_and_split_documents(pdf_dir):
    # Lade PDF-Dateien aus dem angegebenen Verzeichnis
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        raise ValueError(f"Keine PDF-Dateien im Verzeichnis gefunden: {pdf_dir}")

    documents = []
    for pdf_file in pdf_files:
        # Lade jede PDF-Datei
        pdf_loader = PyPDFLoader(pdf_file)
        documents.extend(pdf_loader.load())

    # Teile die geladenen Dokumente in kleinere Abschnitte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store(split_docs, embeddings):
    # Erstelle einen Vektorspeicher aus den geteilten Dokumenten und Einbettungen
    return Chroma.from_documents(split_docs, embeddings)

def initialize_qa_chain(vector_store, openai_api_key):
    # Initialisiere das OpenAI Sprachmodell
    llm = OpenAI(api_key=openai_api_key, max_tokens=200)
    # Erstelle einen Retriever aus dem Vektorspeicher
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Erstelle eine RetrievalQA-Kette
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

def answer_question(qa_chain, question):
    # Verwende die QA-Kette, um die Frage zu beantworten
    return qa_chain.run(question)

def main():
    try:
        # Lade den OpenAI API-Schlüssel
        openai_api_key = load_environment_variables()
        # Verzeichnis, das PDF-Dateien enthält
        pdf_dir = "/home/mehmet/Dokumente/rag_chatbot/pdfs/"
        # Lade und teile die Dokumente
        split_docs = load_and_split_documents(pdf_dir)
        # Erstelle Einbettungen
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Erstelle einen Vektorspeicher
        vector_store = create_vector_store(split_docs, embeddings)
        # Initialisiere die QA-Kette
        qa_chain = initialize_qa_chain(vector_store, openai_api_key)

        while True:
            # Fordere den Benutzer auf, eine Frage zu stellen
            question = input("Stellen Sie eine Frage: ")
            if question.lower() in ["exit", "quit"]:
                break
            # Erhalte die Antwort auf die Frage
            answer = answer_question(qa_chain, question)
            print("Antwort:", answer)
    except Exception as e:
        # Protokolliere alle auftretenden Fehler
        logging.error(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()