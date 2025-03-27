import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title
title = nbf.v4.new_markdown_cell("# Neural Machine Translation: English to French\n\nThis notebook demonstrates the evaluation of different English to French translation models using multiple metrics.")
nb.cells.append(title)

# Install packages
install_packages = nbf.v4.new_code_cell('''# Install required packages
!pip install transformers datasets evaluate sacrebleu
!pip install torch torchvision torchaudio
!pip install sentencepiece
!pip install sacremoses
!pip install bert_score''')
nb.cells.append(install_packages)

# Import packages
imports = nbf.v4.new_code_cell('''from transformers import pipeline
from datasets import load_dataset
import torch
import evaluate
from evaluate import load
import matplotlib.pyplot as plt
import numpy as np''')
nb.cells.append(imports)

# Section 1: Dataset and Models
section1_title = nbf.v4.new_markdown_cell('''## Section 1: Dataset and Models\n\nIn this section, we'll:
1. Create a dataset of 50 English sentences
2. Load 3 different English to French translation models
3. Generate translations for our dataset''')
nb.cells.append(section1_title)

# Create dataset
dataset = nbf.v4.new_code_cell('''# Create our dataset of 50 English sentences
english_sentences = [
    "The cat is sleeping on the windowsill.",
    "I love to read books in the evening.",
    "The weather is beautiful today.",
    "She works at a local restaurant.",
    "We are going to the park tomorrow.",
    "The movie was very interesting.",
    "He speaks French fluently.",
    "The children are playing in the garden.",
    "I need to buy some groceries.",
    "The train arrives at 3 PM.",
    "She likes to drink coffee in the morning.",
    "The museum is closed on Mondays.",
    "We visited Paris last summer.",
    "The book is on the table.",
    "He plays the piano very well.",
    "The restaurant serves delicious food.",
    "I want to learn a new language.",
    "The sun is setting behind the mountains.",
    "She writes beautiful poetry.",
    "We need to clean the house.",
    "The dog is barking at the mailman.",
    "I enjoy listening to classical music.",
    "The store opens at 9 AM.",
    "He is studying for his exams.",
    "The flowers are blooming in spring.",
    "She drives to work every day.",
    "We watched a movie last night.",
    "The birds are singing in the trees.",
    "I need to make a phone call.",
    "The library is quiet and peaceful.",
    "He likes to cook Italian food.",
    "The beach is crowded in summer.",
    "She is learning to swim.",
    "We went shopping yesterday.",
    "The clock is ticking on the wall.",
    "I enjoy taking long walks.",
    "The museum has many paintings.",
    "He plays soccer with his friends.",
    "The restaurant is full of customers.",
    "She reads the newspaper every morning.",
    "We need to fix the computer.",
    "The cat is chasing a mouse.",
    "I want to visit the zoo.",
    "The garden needs watering.",
    "He speaks three languages.",
    "The movie starts at 7 PM.",
    "She likes to dance ballet.",
    "We are planning a vacation.",
    "The birds are flying south.",
    "I need to buy new shoes."
]

# Create reference French translations
french_references = [
    "Le chat dort sur le rebord de la fenêtre.",
    "J'aime lire des livres le soir.",
    "Le temps est beau aujourd'hui.",
    "Elle travaille dans un restaurant local.",
    "Nous allons au parc demain.",
    "Le film était très intéressant.",
    "Il parle couramment le français.",
    "Les enfants jouent dans le jardin.",
    "J'ai besoin d'acheter des provisions.",
    "Le train arrive à 15h.",
    "Elle aime boire du café le matin.",
    "Le musée est fermé le lundi.",
    "Nous avons visité Paris l'été dernier.",
    "Le livre est sur la table.",
    "Il joue très bien du piano.",
    "Le restaurant sert une délicieuse nourriture.",
    "Je veux apprendre une nouvelle langue.",
    "Le soleil se couche derrière les montagnes.",
    "Elle écrit de la belle poésie.",
    "Nous devons nettoyer la maison.",
    "Le chien aboie contre le facteur.",
    "J'aime écouter de la musique classique.",
    "Le magasin ouvre à 9h.",
    "Il étudie pour ses examens.",
    "Les fleurs fleurissent au printemps.",
    "Elle conduit au travail tous les jours.",
    "Nous avons regardé un film hier soir.",
    "Les oiseaux chantent dans les arbres.",
    "Je dois faire un appel téléphonique.",
    "La bibliothèque est calme et paisible.",
    "Il aime cuisiner des plats italiens.",
    "La plage est bondée en été.",
    "Elle apprend à nager.",
    "Nous sommes allés faire des courses hier.",
    "L'horloge tic-tac sur le mur.",
    "J'aime faire de longues promenades.",
    "Le musée a beaucoup de tableaux.",
    "Il joue au football avec ses amis.",
    "Le restaurant est plein de clients.",
    "Elle lit le journal tous les matins.",
    "Nous devons réparer l'ordinateur.",
    "Le chat poursuit une souris.",
    "Je veux visiter le zoo.",
    "Le jardin a besoin d'être arrosé.",
    "Il parle trois langues.",
    "Le film commence à 19h.",
    "Elle aime danser le ballet.",
    "Nous planifions des vacances.",
    "Les oiseaux migrent vers le sud.",
    "J'ai besoin d'acheter de nouvelles chaussures."
]

print(f"Dataset size: {len(english_sentences)} sentences")''')
nb.cells.append(dataset)

# Load models
load_models = nbf.v4.new_code_cell('''# Load three different translation models
model1 = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
model2 = pipeline("translation_en_to_fr", model="facebook/mbart-large-50-many-to-many-mmt")
model3 = pipeline("translation_en_to_fr", model="t5-base")''')
nb.cells.append(load_models)

# Translation function
translate_function = nbf.v4.new_code_cell('''# Function to translate sentences using a model
def translate_sentences(model, sentences):
    translations = []
    for sentence in sentences:
        translated_text = model(sentence)[0]['translation_text']
        translations.append(translated_text)
    return translations

# Generate translations using all three models
translations_model1 = translate_sentences(model1, english_sentences)
translations_model2 = translate_sentences(model2, english_sentences)
translations_model3 = translate_sentences(model3, english_sentences)

print("Translations completed for all three models")''')
nb.cells.append(translate_function)

# Section 2: SACREBLEU Evaluation
section2_title = nbf.v4.new_markdown_cell('''## Section 2: SACREBLEU Evaluation\n\nIn this section, we'll evaluate the translations using the SACREBLEU metric.''')
nb.cells.append(section2_title)

# SACREBLEU evaluation
bleu_eval = nbf.v4.new_code_cell('''# Load SACREBLEU metric
bleu_metric = evaluate.load("sacrebleu")

# Calculate BLEU scores for each model
bleu_score1 = bleu_metric.compute(predictions=translations_model1, references=[french_references])
bleu_score2 = bleu_metric.compute(predictions=translations_model2, references=[french_references])
bleu_score3 = bleu_metric.compute(predictions=translations_model3, references=[french_references])

print("SACREBLEU Scores:")
print(f"Model 1 (Helsinki-NLP): {bleu_score1['score']:.2f}")
print(f"Model 2 (mBART): {bleu_score2['score']:.2f}")
print(f"Model 3 (T5): {bleu_score3['score']:.2f}")''')
nb.cells.append(bleu_eval)

# Section 3: BERTScore Evaluation
section3_title = nbf.v4.new_markdown_cell('''## Section 3: BERTScore Evaluation\n\nIn this section, we'll evaluate the translations using the BERTScore metric.''')
nb.cells.append(section3_title)

# BERTScore evaluation
bert_eval = nbf.v4.new_code_cell('''# Load BERTScore metric
bert_metric = load("bertscore")

# Calculate BERTScore for each model
bert_results1 = bert_metric.compute(predictions=translations_model1, references=french_references, lang="fr")
bert_results2 = bert_metric.compute(predictions=translations_model2, references=french_references, lang="fr")
bert_results3 = bert_metric.compute(predictions=translations_model3, references=french_references, lang="fr")

# Calculate average F1 scores
avg_f1_1 = sum(bert_results1['f1']) / len(bert_results1['f1'])
avg_f1_2 = sum(bert_results2['f1']) / len(bert_results2['f1'])
avg_f1_3 = sum(bert_results3['f1']) / len(bert_results3['f1'])

print("BERTScore F1 Scores:")
print(f"Model 1 (Helsinki-NLP): {avg_f1_1:.4f}")
print(f"Model 2 (mBART): {avg_f1_2:.4f}")
print(f"Model 3 (T5): {avg_f1_3:.4f}")''')
nb.cells.append(bert_eval)

# Section 4: Visualization
section4_title = nbf.v4.new_markdown_cell('''## Section 4: Visualization\n\nIn this section, we'll create visualizations to compare the performance of the three models using both metrics.''')
nb.cells.append(section4_title)

# Visualization
visualization = nbf.v4.new_code_cell('''# Normalize BLEU scores to 0-1 range for comparison
bleu1_norm = bleu_score1['score'] / 100
bleu2_norm = bleu_score2['score'] / 100
bleu3_norm = bleu_score3['score'] / 100

# Set up the comparison data
models = ['Helsinki-NLP', 'mBART', 'T5']
bleu_scores = [bleu1_norm, bleu2_norm, bleu3_norm]
bert_scores = [avg_f1_1, avg_f1_2, avg_f1_3]

# Set width of bars
barWidth = 0.3
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]

# Create the bars
plt.figure(figsize=(12, 6))
plt.bar(r1, bleu_scores, width=barWidth, label='SACREBLEU (normalized)')
plt.bar(r2, bert_scores, width=barWidth, label='BERTScore F1')

# Add labels and title
plt.xlabel('Translation Models')
plt.ylabel('Score (0-1 scale)')
plt.title('Comparison of Translation Models using SACREBLEU and BERTScore')
plt.xticks([r + barWidth/2 for r in range(len(models))], models)
plt.ylim(0, 1.0)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()''')
nb.cells.append(visualization)

# Analysis
analysis = nbf.v4.new_markdown_cell('''### Analysis of Results\n\nThe visualization shows the performance of three different translation models evaluated using both SACREBLEU and BERTScore metrics:\n\n1. **SACREBLEU Scores**:\n   - Measures exact word matches between translations and references\n   - Higher scores indicate more word-level accuracy\n   - Normalized to 0-1 scale for easier comparison\n\n2. **BERTScore F1**:\n   - Measures semantic similarity between translations and references\n   - Higher scores indicate better meaning preservation\n   - Already on a 0-1 scale\n\nThe comparison reveals:\n- Which model performs best according to each metric\n- How the models differ in terms of word-level accuracy vs. semantic accuracy\n- The relative strengths and weaknesses of each model''')
nb.cells.append(analysis)

# Write the notebook to a file
with open('nmt_notebook_updated.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 