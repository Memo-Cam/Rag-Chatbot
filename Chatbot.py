import os
import glob
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#os.environ["OPENAI_API_KEY"] = "your_openai_api_key" #ggf auskommentieren bei Windows. 

logging.basicConfig(level=logging.DEBUG)  # Temporär auf DEBUG setzen für detailliertes Logging

def load_environment_variables():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY Umgebungsvariable nicht gesetzt")
    return openai_api_key

def load_and_split_documents(pdf_dir):
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        raise ValueError(f"Keine PDF-Dateien im Verzeichnis gefunden: {pdf_dir}")

    documents = []
    for pdf_file in pdf_files:
        pdf_loader = PyPDFLoader(pdf_file)
        documents.extend(pdf_loader.load())

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store(split_docs, embeddings):
    return Chroma.from_documents(split_docs, embeddings)

def initialize_qa_chain(vector_store, openai_api_key):
    system_prompt = (
        "Du bist ein reiner Informationsassistent, der ausschließlich auf den bereitgestellten Dokumenten basiert. "
        "Dein einziges Wissen stammt aus den bereitgestellten Dokumenten über die Wahlprogramme der Parteien. "
        "Du hast kein Wissen außerhalb dieser Dokumente und kannst keine Informationen hinzufügen. "
        "Beantworte die Fragen ausschließlich basierend auf den Inhalten dieser Dokumente. "
        "Falls die Informationen nicht vorhanden sind, antworte mit: 'Dazu liegen keine Informationen vor.' "
        "Falls du die Antworten nicht kennst, antworte mit: 'Dazu liegen keine Informationen vor.' "
        "Antworte kurz und präzise. "
        "Antworte nur auf Deutsch. "
    )
    llm = OpenAI(api_key=openai_api_key, max_tokens=200, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(system_prompt + "\n\nFrage: {question}\nAntwort:")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Wechsel zu "map_reduce" für bessere Handhabung der Antworten
        retriever=retriever,
        return_source_documents=True  # Muss True sein, damit im Zitatmodus die Quellen zurückgegeben werden
    )
    return qa_chain

def answer_question(qa_chain, question, quote_mode=False):
    parties = ["FDP", "SPD", "CDU", "AFD", "GRÜNE", "BSW", "Die Linke"]
    responses = []

    # Falls keine Partei explizit in der Frage genannt wird, frage für jede Partei einzeln nach
    if not any(party.lower() in question.lower() for party in parties):
        for party in parties:
            modified_question = f"Was sagt die {party} dazu? {question}"
            try:
                result = qa_chain(modified_question)
                answer = result.get("result", "")
                sources = result.get("source_documents", []) if quote_mode else []
                if "Dazu liegen keine Informationen vor." not in answer:
                    res = f"**{party}**: {answer}"
                    # Zusätzliche Quellinfos nur im Zitatmodus
                    if quote_mode and sources:
                        for doc in sources:
                            page = doc.metadata.get("page", "unbekannt")
                            snippet = doc.page_content.strip().replace("\n", " ")
                            res += f"\n - {party} Wahlprogramm: {snippet[:200]} (Seite: {page})"
                    responses.append(res)
                else:
                    responses.append(f"**{party}**: Dazu liegen keine Informationen vor.")
            except Exception as e:
                logging.error(f"Fehler bei der Beantwortung der Frage für {party}: {e}")
                responses.append(f"**{party}**: Fehler bei der Antwortgenerierung.")
        return "\n\n".join(responses)

    # Falls bereits eine Partei in der Frage genannt wurde, nutze die Frage direkt
    try:
        result = qa_chain(question)
        answer = result.get("result", "")
        sources = result.get("source_documents", []) if quote_mode else []
        logging.debug(f"Frage: {question}")
        logging.debug(f"Antwort: {answer}")

        if "Dazu liegen keine Informationen vor." in answer:
            return "Dazu liegen keine Informationen vor."
        if quote_mode and sources:
            source_text = ""
            for doc in sources:
                page = doc.metadata.get("page", "unbekannt")
                snippet = doc.page_content.strip().replace("\n", " ")
                source_text += f"\n - Wahlprogramm: {snippet[:200]} (Seite: {page})"
            return answer + source_text
        return answer
    except Exception as e:
        logging.error(f"Fehler bei der Beantwortung der Frage: {e}")
        return "Dazu liegen keine Informationen vor."

def main():
    try:
        openai_api_key = load_environment_variables()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(script_dir, "pdfs")        
        split_docs = load_and_split_documents(pdf_dir)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = create_vector_store(split_docs, embeddings)
        qa_chain = initialize_qa_chain(vector_store, openai_api_key)

        while True:
            question = input("Stellen Sie eine Frage: ")
            if question.lower() in ["exit", "quit"]:
                break
            answer = answer_question(qa_chain, question, quote_mode=False)
            print("Antwort:", answer)
    except Exception as e:
        logging.error(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()