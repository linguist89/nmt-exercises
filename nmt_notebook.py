#!/usr/bin/env python
# coding: utf-8

# <h1> Neural Machine Translation

# In[ ]:


# using hugging face translation pipeline


# In[1]:


get_ipython().system('pip install transformers datasets evaluate sacrebleu')
get_ipython().system('pip install torch torchvision torchaudio')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install sacremoses')
get_ipython().system('pip install bert_score')


# In[1]:


from transformers import pipeline
from datasets import load_dataset
import torch
import evaluate
from evaluate import load


# In[ ]:


# Let's say you want to translate a sentence from English to French. Try running the default model T5-base for your translation


# In[2]:


en_fr_translator = pipeline("translation_en_to_fr")
en_fr_translator("How old are you?")
## [{'translation_text': ' quel âge êtes-vous?'}]


# In[3]:


# Now, try loading a specific model for your languages. For more models, check the webpage: https://huggingface.co/Helsinki-NLP


# In[4]:


translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")


# In[5]:


# Load your own corpus and explore it


# In[6]:


my_corpus = [ "This is a test.", "This is another test."]


# In[7]:


# Build a function that takes your corpus as input and returns their translation


# In[8]:


def translate_sentences(my_corpus):
    translations = []  
    for sentence in my_corpus:
        translated_text = translator(sentence)[0]['translation_text'] 
        translations.append(translated_text)  
    return translations 


# In[9]:


target_sentences = translate_sentences(my_corpus)


# In[10]:


print(target_sentences)


# In[ ]:


# In order to evaluate your translations, you are going to need a reference translations (the correct ones). 


# In[ ]:


# Evaluate the translation using the Bleu score


# In[11]:


metric = evaluate.load("sacrebleu")


# In[21]:


reference_translations = ["C'est un test.", "C'est un autre test."]


# In[22]:


predicted_translation = target_sentences


# In[23]:


bleu_score = metric.compute(predictions=predicted_translation, references=reference_translations)


# In[24]:


print("BLEU Score:", bleu_score["score"])


# In[25]:


# Evaluate the translation using the BERT score


# In[26]:


bertscore = load("bertscore")


# In[27]:


results = bertscore.compute(predictions=target_sentences, references=reference_translations, lang="fr")


# In[29]:


results


# In[28]:


# Access f1 scores directly from results dictionary
average_f1_score = sum(results['f1']) / len(results['f1'])  # Calculate the average F1 score
print("BERTScore F1:", average_f1_score)


# In[ ]:


# Compare a single sentence using Bleu and BERT. What do you observe?


# In[ ]:


# Experiment with translations of other languages you know.


# In[ ]:


# Are the accuracy scores bidirectional with the languages?

