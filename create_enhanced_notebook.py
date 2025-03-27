import nbformat as nbf
import json

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and Introduction
intro_md = """# Neural Machine Translation: Model Comparison Across Topics

In this notebook, we'll explore how different Neural Machine Translation (NMT) models perform across various topics. We'll learn how to:
1. Load and prepare different types of translation data
2. Use the Hugging Face pipeline for translation
3. Evaluate translations using multiple metrics
4. Visualize and analyze model performance
5. Compare translations across different topics

Let's start by setting up our environment and loading the necessary libraries."""

nb['cells'].append(nbf.v4.new_markdown_cell(intro_md))

# Imports
imports_md = """First, we need to install and import the required libraries. We'll use:
- `transformers` for the translation models
- `pandas` for data handling
- `matplotlib` and `seaborn` for visualization
- `sacrebleu` and `bert_score` for evaluation metrics"""

nb['cells'].append(nbf.v4.new_markdown_cell(imports_md))

imports_code = """# Install required packages if not already installed
!pip install transformers datasets evaluate sacrebleu
!pip install torch torchvision torchaudio
!pip install sentencepiece
!pip install sacremoses
!pip install bert_score

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from bert_score import score
from sacrebleu import corpus_bleu
from tqdm import tqdm

# Set style for better visualizations
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will apply seaborn's styling"""

nb['cells'].append(nbf.v4.new_code_cell(imports_code))

# Dataset Loading
dataset_md = """## 1. Loading and Preparing Our Datasets

We'll be working with three different datasets, each focusing on a specific topic:
1. General translations (everyday conversations and situations)
2. Political translations (government, policies, elections)
3. Sports translations (matches, tournaments, athletes)

Let's first create a function to load our CSV files and then combine them into a single dataset."""

nb['cells'].append(nbf.v4.new_markdown_cell(dataset_md))

dataset_code = """# Function to load a dataset from a CSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Load each dataset
print("Loading general translations...")
general_df = load_dataset('data/general_translations.csv')

print("Loading political translations...")
politics_df = load_dataset('data/political_translations.csv')

print("Loading sports translations...")
sports_df = load_dataset('data/sports_translations.csv')

# Add topic labels to each dataset
general_df['topic'] = 'general'
politics_df['topic'] = 'politics'
sports_df['topic'] = 'sports'

# Combine all datasets
all_data = pd.concat([general_df, politics_df, sports_df], ignore_index=True)

# Display dataset information
print("\\nDataset Overview:")
print(f"Total samples: {len(all_data)}")
print("\\nSamples per topic:")
print(all_data['topic'].value_counts())

# Display a few examples from each topic
print("\\nExample from each topic:")
for topic in ['general', 'politics', 'sports']:
    example = all_data[all_data['topic'] == topic].iloc[0]
    print(f"\\n{topic.capitalize()} Topic:")
    print(f"Source: {example['source_text']}")
    print(f"Target: {example['target_text']}")"""

nb['cells'].append(nbf.v4.new_code_cell(dataset_code))

# Model Initialization
model_md = """## 2. Setting Up Our Translation Models

We'll compare three different translation models:
1. T5-small: A smaller, faster version of the T5 model
2. T5-base: The standard T5 model with better performance
3. Helsinki-NLP: A specialized model for English to French translation

Let's initialize these models using the Hugging Face pipeline. The pipeline makes it easy to use pre-trained models for various tasks."""

nb['cells'].append(nbf.v4.new_markdown_cell(model_md))

model_code = """# Initialize translation pipelines
print("Loading T5-small...")
t5_small = pipeline("translation", model="t5-small")

print("Loading T5-base...")
t5_base = pipeline("translation", model="t5-base")

print("Loading Helsinki-NLP model...")
helsinki = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Store models in a dictionary for easier access
models = {
    'T5-small': t5_small,
    'T5-base': t5_base,
    'Helsinki-NLP': helsinki
}

# Test each model with a simple sentence
test_sentence = "Hello, how are you?"
print("\\nTesting each model with a simple sentence:")
for model_name, model in models.items():
    # Add task prefix for T5 models
    if model_name.startswith('T5'):
        text = f"translate English to French: {test_sentence}"
    else:
        text = test_sentence
    
    result = model(text)[0]['translation_text']
    print(f"\\n{model_name}:")
    print(f"Input: {test_sentence}")
    print(f"Output: {result}")"""

nb['cells'].append(nbf.v4.new_code_cell(model_code))

# Translation Process
translation_md = """## 3. Translating Our Datasets

Let's translate our datasets step by step. We'll:
1. First translate a small subset to test our setup
2. Then translate the full dataset for each topic
3. Store the translations for evaluation"""

nb['cells'].append(nbf.v4.new_markdown_cell(translation_md))

translation_code = """# First, let's test with a small subset
print("Testing translation with a small subset...")
test_subset = all_data.head(3)
translations = {}

for model_name, model in models.items():
    print(f"\\nTranslating with {model_name}...")
    model_translations = []
    
    for _, row in test_subset.iterrows():
        # Add task prefix for T5 models
        if model_name.startswith('T5'):
            text = f"translate English to French: {row['source_text']}"
        else:
            text = row['source_text']
        
        result = model(text)[0]['translation_text']
        model_translations.append(result)
        print(f"Source: {row['source_text']}")
        print(f"Translation: {result}")
    
    translations[model_name] = model_translations

# Now let's translate the full dataset
print("\\nTranslating full dataset...")
full_translations = {}

for model_name, model in models.items():
    print(f"\\nTranslating with {model_name}...")
    model_translations = []
    
    for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc=f"Translating with {model_name}"):
        if model_name.startswith('T5'):
            text = f"translate English to French: {row['source_text']}"
        else:
            text = row['source_text']
        
        result = model(text)[0]['translation_text']
        model_translations.append(result)
    
    full_translations[model_name] = model_translations

print("\\nTranslation complete!")"""

nb['cells'].append(nbf.v4.new_code_cell(translation_code))

# BLEU Evaluation
bleu_md = """## 4. Evaluating with BLEU Score

Now that we have our translations, let's evaluate them using the BLEU score. We'll:
1. Calculate BLEU scores for our test subset
2. Calculate BLEU scores for the full dataset
3. Compare scores across models and topics"""

nb['cells'].append(nbf.v4.new_markdown_cell(bleu_md))

bleu_code = """# First, let's evaluate our test subset
print("Evaluating test subset with BLEU...")
test_bleu_scores = {}

for model_name, model_translations in translations.items():
    bleu_score = corpus_bleu(model_translations, [test_subset['target_text'].tolist()]).score
    test_bleu_scores[model_name] = bleu_score
    print(f"{model_name} BLEU Score: {bleu_score:.2f}")

# Now evaluate the full dataset
print("\\nEvaluating full dataset with BLEU...")
full_bleu_scores = {}

for model_name, model_translations in full_translations.items():
    bleu_score = corpus_bleu(model_translations, [all_data['target_text'].tolist()]).score
    full_bleu_scores[model_name] = bleu_score
    print(f"{model_name} BLEU Score: {bleu_score:.2f}")

# Calculate BLEU scores by topic
print("\\nCalculating BLEU scores by topic...")
topic_bleu_scores = {}

for topic in ['general', 'politics', 'sports']:
    topic_data = all_data[all_data['topic'] == topic]
    topic_indices = topic_data.index
    
    print(f"\\n{topic.capitalize()} Topic BLEU Scores:")
    for model_name, model_translations in full_translations.items():
        topic_translations = [model_translations[i] for i in topic_indices]
        bleu_score = corpus_bleu(topic_translations, [topic_data['target_text'].tolist()]).score
        topic_bleu_scores[(model_name, topic)] = bleu_score
        print(f"{model_name}: {bleu_score:.2f}")"""

nb['cells'].append(nbf.v4.new_code_cell(bleu_code))

# BERTScore Evaluation
bert_md = """## 5. Evaluating with BERTScore

Now let's evaluate our translations using BERTScore, which provides a different perspective on translation quality. We'll:
1. Calculate BERTScore for our test subset
2. Calculate BERTScore for the full dataset
3. Compare scores across models and topics"""

nb['cells'].append(nbf.v4.new_markdown_cell(bert_md))

bert_code = """# First, evaluate our test subset
print("Evaluating test subset with BERTScore...")
test_bert_scores = {}

for model_name, model_translations in translations.items():
    P, R, F1 = score(model_translations, test_subset['target_text'].tolist(), lang='fr', verbose=False)
    bert_score = F1.mean().item()
    test_bert_scores[model_name] = bert_score
    print(f"{model_name} BERTScore: {bert_score:.2f}")

# Now evaluate the full dataset
print("\\nEvaluating full dataset with BERTScore...")
full_bert_scores = {}

for model_name, model_translations in full_translations.items():
    P, R, F1 = score(model_translations, all_data['target_text'].tolist(), lang='fr', verbose=False)
    bert_score = F1.mean().item()
    full_bert_scores[model_name] = bert_score
    print(f"{model_name} BERTScore: {bert_score:.2f}")

# Calculate BERTScore by topic
print("\\nCalculating BERTScore by topic...")
topic_bert_scores = {}

for topic in ['general', 'politics', 'sports']:
    topic_data = all_data[all_data['topic'] == topic]
    topic_indices = topic_data.index
    
    print(f"\\n{topic.capitalize()} Topic BERTScore:")
    for model_name, model_translations in full_translations.items():
        topic_translations = [model_translations[i] for i in topic_indices]
        P, R, F1 = score(topic_translations, topic_data['target_text'].tolist(), lang='fr', verbose=False)
        bert_score = F1.mean().item()
        topic_bert_scores[(model_name, topic)] = bert_score
        print(f"{model_name}: {bert_score:.2f}")"""

nb['cells'].append(nbf.v4.new_code_cell(bert_code))

# Combine Results
results_md = """## 6. Combining and Analyzing Results

Now let's combine all our evaluation results into a single DataFrame for easier analysis."""

nb['cells'].append(nbf.v4.new_markdown_cell(results_md))

results_code = """# Create a DataFrame with all results
results = []
for model_name in models.keys():
    for topic in ['general', 'politics', 'sports']:
        results.append({
            'model': model_name,
            'topic': topic,
            'bleu_score': topic_bleu_scores[(model_name, topic)],
            'bert_score': topic_bert_scores[(model_name, topic)]
        })

results_df = pd.DataFrame(results)

# Display the results
print("Evaluation Results:")
print(results_df)"""

nb['cells'].append(nbf.v4.new_code_cell(results_code))

# Visualizations
viz_md = """## 7. Visualizing the Results

Let's create visualizations to better understand how the models perform across different topics. We'll create:
1. Bar plots comparing BLEU scores
2. Bar plots comparing BERTScore
3. A heatmap showing overall performance"""

nb['cells'].append(nbf.v4.new_markdown_cell(viz_md))

viz_code = """# Set figure size for all plots
plt.rcParams['figure.figsize'] = [12, 6]

# 1. BLEU Scores by Model and Topic
plt.figure()
sns.barplot(data=results_df, x='model', y='bleu_score', hue='topic')
plt.title('BLEU Scores by Model and Topic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. BERTScore by Model and Topic
plt.figure()
sns.barplot(data=results_df, x='model', y='bert_score', hue='topic')
plt.title('BERTScore by Model and Topic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Heatmap of Model Performance
pivot_bleu = results_df.pivot(index='model', columns='topic', values='bleu_score')
plt.figure()
sns.heatmap(pivot_bleu, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('BLEU Score Heatmap')
plt.tight_layout()
plt.show()"""

nb['cells'].append(nbf.v4.new_code_cell(viz_code))

# Topic Analysis
topic_md = """## 8. Analyzing Topic-Specific Performance

Let's analyze how each model performs on different topics. We'll look at:
1. Best performing model for each topic
2. Model rankings by both metrics
3. Topic-specific strengths and weaknesses"""

nb['cells'].append(nbf.v4.new_markdown_cell(topic_md))

topic_code = """def analyze_topic_performance(topic):
    topic_data = results_df[results_df['topic'] == topic]
    
    print(f"\\n=== Performance Analysis for {topic.capitalize()} Topic ===")
    print("\\nBest Model by BLEU Score:")
    best_bleu = topic_data.loc[topic_data['bleu_score'].idxmax()]
    print(f"Model: {best_bleu['model']}, Score: {best_bleu['bleu_score']:.2f}")
    
    print("\\nBest Model by BERTScore:")
    best_bert = topic_data.loc[topic_data['bert_score'].idxmax()]
    print(f"Model: {best_bert['model']}, Score: {best_bert['bert_score']:.2f}")
    
    print("\\nModel Rankings:")
    print("BLEU Score Rankings:")
    print(topic_data.sort_values('bleu_score', ascending=False)[['model', 'bleu_score']])
    print("\\nBERTScore Rankings:")
    print(topic_data.sort_values('bert_score', ascending=False)[['model', 'bert_score']])

# Analyze each topic
for topic in ['general', 'politics', 'sports']:
    analyze_topic_performance(topic)"""

nb['cells'].append(nbf.v4.new_code_cell(topic_code))

# Example Translations
examples_md = """## 9. Looking at Example Translations

Finally, let's examine some actual translations from each model to qualitatively assess their performance. We'll look at:
1. How well they handle topic-specific vocabulary
2. The accuracy of their translations
3. Any patterns in their strengths and weaknesses"""

nb['cells'].append(nbf.v4.new_markdown_cell(examples_md))

examples_code = """def show_examples(topic, num_examples=3):
    topic_data = all_data[all_data['topic'] == topic]
    examples = topic_data.sample(n=num_examples)
    
    print(f"\\n=== Example Translations for {topic.capitalize()} Topic ===")
    for idx, row in examples.iterrows():
        print(f"\\nExample {idx + 1}:")
        print(f"Source: {row['source_text']}")
        print(f"Reference: {row['target_text']}")
        
        # Get translations from each model
        for model_name, model in models.items():
            # Add task prefix for T5 models
            if model_name.startswith('T5'):
                text = f"translate English to French: {row['source_text']}"
            else:
                text = row['source_text']
            
            result = model(text)[0]['translation_text']
            print(f"{model_name}: {result}")

# Show examples for each topic
for topic in ['general', 'politics', 'sports']:
    show_examples(topic)"""

nb['cells'].append(nbf.v4.new_code_cell(examples_code))

# Conclusion
conclusion_md = """## 10. Summary and Conclusions

In this notebook, we've:
1. Loaded and prepared different types of translation data
2. Set up three different translation models
3. Translated the datasets
4. Evaluated their performance using BLEU and BERTScore
5. Visualized and analyzed the results
6. Examined example translations

Key takeaways:
- Different models may perform better on different topics
- Using multiple evaluation metrics gives us a more complete picture
- The choice of model might depend on the specific use case
- Topic-specific training data might improve performance

Would you like to experiment with:
1. Different models?
2. Different topics?
3. Different evaluation metrics?
4. Different visualization approaches?"""

nb['cells'].append(nbf.v4.new_markdown_cell(conclusion_md))

# Write the notebook to a file
with open('nmt_notebook_updated.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 