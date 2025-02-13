import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from Chatbot import (
    load_and_split_documents,
    create_vector_store,
    initialize_qa_chain,
    answer_question,
    load_environment_variables,
    OpenAIEmbeddings
)

class ChatbotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chatbot")
        self.geometry("800x600")

        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # Verwenden eines modernen Themas

        self.pdf_dir = "/home/mehmet/Dokumente/rag_chatbot/chatbot/pdfs/"# Pfad zu den PDF-Dateien bitte Anpassen
        self.openai_api_key = load_environment_variables()
        self.split_docs = load_and_split_documents(self.pdf_dir)
        self.vector_store = create_vector_store(self.split_docs, OpenAIEmbeddings(openai_api_key=self.openai_api_key))
        self.qa_chain = initialize_qa_chain(self.vector_store, self.openai_api_key)

        # Variable für den Zitatmodus
        self.quote_mode_var = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        self.chat_display = scrolledtext.ScrolledText(self, wrap=tk.WORD, state='disabled', bg="#f0f0f0", font=("Arial", 12))
        self.chat_display.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.question_frame = ttk.Frame(self)
        self.question_frame.pack(pady=10, padx=10, fill=tk.X)

        self.question_entry = ttk.Entry(self.question_frame, font=("Arial", 12))
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.answer_button = ttk.Button(self.question_frame, text="Antwort erhalten", command=self.get_answer)
        self.answer_button.pack(side=tk.RIGHT)

        # Checkbutton für den Zitatmodus
        self.quote_mode_cb = ttk.Checkbutton(self, text="Zitatmodus", variable=self.quote_mode_var)
        self.quote_mode_cb.pack(pady=(0,10))

        self.configure_tags()

    def get_answer(self):
        question = self.question_entry.get().strip()
        if not question:
            return  # Leere Eingaben ignorieren
        if question.lower() in ["exit", "quit"]:
            self.quit()
        else:
            self.display_message(f"Sie: {question}\n", "user")
            # Übergibt den Status des Zitatmodus an answer_question
            answer = answer_question(self.qa_chain, question, quote_mode=self.quote_mode_var.get())

            # Überprüfen, ob die Antwort die Standardnachricht ist
            if "Dazu liegen keine Informationen vor." in answer:
                display_text = "Chatbot: Dazu liegen keine Informationen vor.\n"
            else:
                display_text = f"Chatbot: {answer}\n"

            self.display_message(display_text, "bot")
            self.question_entry.delete(0, tk.END)

    def display_message(self, message, sender):
        self.chat_display.config(state='normal')
        if sender == "user":
            self.chat_display.insert(tk.END, message, "user")
        else:
            self.chat_display.insert(tk.END, message, "bot")
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)

    def configure_tags(self):
        self.chat_display.tag_config("user", foreground="blue", font=("Arial", 12, "bold"))
        self.chat_display.tag_config("bot", foreground="green", font=("Arial", 12))

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()