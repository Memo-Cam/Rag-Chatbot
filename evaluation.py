import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Lade die Goldstandard-Antworten
with open("evaluation/GoldenErgebnisse.json", "r") as f:
    gold_standard = json.load(f)

# Lade die ChatGPT-Ergebnisse
with open("evaluation/ChatGPTErgebnisse.json", "r") as f:
    chatgpt_results = json.load(f)

# Lade die Rag-Ergebnisse
with open("evaluation/RagErgebnisse.json", "r") as f:
    rag_results = json.load(f)

# Lade die ChatGPT35-Ergebnisse
with open("evaluation/ChatGPT35.json", "r") as f:
    chatgpt35_results = json.load(f)

# Bereite die Daten für die Metrikberechnung vor
gold_answers = list(gold_standard.values())
chatgpt_answers = list(chatgpt_results.values())
rag_answers = list(rag_results.values())
chatgpt35_answers = list(chatgpt35_results.values())

# Funktion zur Berechnung der Cosine Similarity
def calculate_similarity(gold, result):
    vectorizer = TfidfVectorizer().fit_transform([gold, result])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Beispiel für die Berechnung des BLEU-Scores mit Glättung
smoothing_function = SmoothingFunction().method1

# Funktion zur Berechnung des BLEU-Scores
def calculate_bleu(gold, result):
    return sentence_bleu([gold.split()], result.split(), smoothing_function=smoothing_function)

# Funktion zur Berechnung des ROUGE-Scores
def calculate_rouge(gold, result):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gold, result)
    return scores

# Berechnung der Metriken für ChatGPT-Ergebnisse
chatgpt_similarities = [calculate_similarity(g, r) for g, r in zip(gold_answers, chatgpt_answers)]
chatgpt_avg_similarity = sum(chatgpt_similarities) / len(chatgpt_similarities)

chatgpt_bleu_scores = [calculate_bleu(g, r) for g, r in zip(gold_answers, chatgpt_answers)]
chatgpt_avg_bleu = sum(chatgpt_bleu_scores) / len(chatgpt_bleu_scores)

chatgpt_rouge_scores = [calculate_rouge(g, r) for g, r in zip(gold_answers, chatgpt_answers)]
chatgpt_avg_rouge1 = sum([score['rouge1'].fmeasure for score in chatgpt_rouge_scores]) / len(chatgpt_rouge_scores)
chatgpt_avg_rougeL = sum([score['rougeL'].fmeasure for score in chatgpt_rouge_scores]) / len(chatgpt_rouge_scores)

# Berechnung der Metriken für Rag-Ergebnisse
rag_similarities = [calculate_similarity(g, r) for g, r in zip(gold_answers, rag_answers)]
rag_avg_similarity = sum(rag_similarities) / len(rag_similarities)

rag_bleu_scores = [calculate_bleu(g, r) for g, r in zip(gold_answers, rag_answers)]
rag_avg_bleu = sum(rag_bleu_scores) / len(rag_bleu_scores)

rag_rouge_scores = [calculate_rouge(g, r) for g, r in zip(gold_answers, rag_answers)]
rag_avg_rouge1 = sum([score['rouge1'].fmeasure for score in rag_rouge_scores]) / len(rag_rouge_scores)
rag_avg_rougeL = sum([score['rougeL'].fmeasure for score in rag_rouge_scores]) / len(rag_rouge_scores)

# Berechnung der Metriken für ChatGPT35-Ergebnisse
chatgpt35_similarities = [calculate_similarity(g, r) for g, r in zip(gold_answers, chatgpt35_answers)]
chatgpt35_avg_similarity = sum(chatgpt35_similarities) / len(chatgpt35_similarities)

chatgpt35_bleu_scores = [calculate_bleu(g, r) for g, r in zip(gold_answers, chatgpt35_answers)]
chatgpt35_avg_bleu = sum(chatgpt35_bleu_scores) / len(chatgpt35_bleu_scores)

chatgpt35_rouge_scores = [calculate_rouge(g, r) for g, r in zip(gold_answers, chatgpt35_answers)]
chatgpt35_avg_rouge1 = sum([score['rouge1'].fmeasure for score in chatgpt35_rouge_scores]) / len(chatgpt35_rouge_scores)
chatgpt35_avg_rougeL = sum([score['rougeL'].fmeasure for score in chatgpt35_rouge_scores]) / len(chatgpt35_rouge_scores)

# Speichere die Ergebnisse in einer JSON-Datei
results = {
    "ChatGPT": {
        "average_similarity": chatgpt_avg_similarity,
        "average_bleu": chatgpt_avg_bleu,
        "average_rouge1": chatgpt_avg_rouge1,
        "average_rougeL": chatgpt_avg_rougeL
    },
    "Rag": {
        "average_similarity": rag_avg_similarity,
        "average_bleu": rag_avg_bleu,
        "average_rouge1": rag_avg_rouge1,
        "average_rougeL": rag_avg_rougeL
    },
    "ChatGPT35": {
        "average_similarity": chatgpt35_avg_similarity,
        "average_bleu": chatgpt35_avg_bleu,
        "average_rouge1": chatgpt35_avg_rouge1,
        "average_rougeL": chatgpt35_avg_rougeL
    }
}

output_file = "evaluation_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Ergebnisse wurden in {output_file} gespeichert.")
print(f"ChatGPT - Average Similarity: {chatgpt_avg_similarity}")
print(f"ChatGPT - Average BLEU: {chatgpt_avg_bleu}")
print(f"ChatGPT - Average ROUGE-1: {chatgpt_avg_rouge1}")
print(f"ChatGPT - Average ROUGE-L: {chatgpt_avg_rougeL}")
print(f"Rag - Average Similarity: {rag_avg_similarity}")
print(f"Rag - Average BLEU: {rag_avg_bleu}")
print(f"Rag - Average ROUGE-1: {rag_avg_rouge1}")
print(f"Rag - Average ROUGE-L: {rag_avg_rougeL}")
print(f"ChatGPT35 - Average Similarity: {chatgpt35_avg_similarity}")
print(f"ChatGPT35 - Average BLEU: {chatgpt35_avg_bleu}")
print(f"ChatGPT35 - Average ROUGE-1: {chatgpt35_avg_rouge1}")
print(f"ChatGPT35 - Average ROUGE-L: {chatgpt35_avg_rougeL}")