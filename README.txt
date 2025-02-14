# RAG-basierter Wahlprogramm-Chatbot

Dieser Chatbot verwendet die `langchain`-Bibliothek, um Wahlprogramme von Parteien aus PDF-Dokumenten zu extrahieren, sie zu analysieren und auf Fragen basierend auf den Inhalten der Dokumente zu antworten. Der Chatbot nutzt OpenAI GPT für die Beantwortung von Fragen und ist auf Deutsch konzipiert.

## Funktionen
- Lädt Wahlprogramme von Parteien aus PDFs.
- Teilt die Dokumente in handliche Abschnitte auf.
- Erstellt einen Vektorstore mit den Dokumenten.
- Beantwortet Fragen basierend auf den Wahlprogrammen der Parteien.
- Zeigt Quellen (Seitenzahl und Ausschnitte) im Zitatmodus an.

## Installation


##Erstelle eine virtuelle Umgebung (optional)
python3 -m venv venv
source venv/bin/activate  # für Unix/macOS
venv\Scripts\activate     # für Windows

##installiere die benötigten Abhängigkeiten
pip install -r requirements.txt

##setze den OpenAI Schlüssel
export OPENAI_API_KEY= "dein_api_schluessel" # für Unix/macOS
set OPENAI_API_KEY="dein_api_schluessel"     # für Windows

## Die Zeile os.environ["OPENAI_API_KEY"] = "your_openai_api_key" ggf auskommentieren bei Windows und den key dort einfügen


##Stelle sicher, dass das Verzeichnis mit den PDF-Dokumenten korrekt angegeben wurde.



##Führe das Skript aus:

python gui.py


##Um die ergebnisse inkl. Zitat zu bekommen den Zitatmodus einschalten



