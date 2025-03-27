import json

# Load the notebook
with open('nmt_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create the new markdown cell with the explanation
new_markdown_cell = {
    "cell_type": "markdown",
    "id": "metric-explanation",
    "metadata": {},
    "source": [
        "## Understanding Translation Evaluation Metrics: SACREBLEU and BERTScore\n",
        "\n",
        "Translation quality metrics help us evaluate how good a machine translation is compared to a human reference. Let's understand the two main metrics we're using:\n",
        "\n",
        "### SACREBLEU: The Traditional Word-Matching Approach\n",
        "\n",
        "SACREBLEU is a standardized implementation of BLEU (Bilingual Evaluation Understudy), which has been the industry standard for years.\n",
        "\n",
        "#### How SACREBLEU Works (with Analogies)\n",
        "\n",
        "1. **N-gram Matching**: SACREBLEU counts how many word sequences in your translation match those in the reference translation.\n",
        "   \n",
        "   **Analogy**: Imagine comparing two recipes. SACREBLEU checks how many exact ingredient combinations (like \"salt and pepper\" or \"olive oil and garlic\") appear in both recipes.\n",
        "\n",
        "2. **Brevity Penalty**: It penalizes translations that are too short.\n",
        "   \n",
        "   **Analogy**: Like a teacher deducting points if your essay is only half a page when it should be two pages, even if what you wrote was correct.\n",
        "\n",
        "3. **Tokenization**: SACREBLEU uses standardized tokenization methods to ensure consistency.\n",
        "   \n",
        "   **Analogy**: Before comparing two documents, making sure both use the same formatting rules (like how to handle punctuation).\n",
        "\n",
        "#### Limitations\n",
        "\n",
        "- **Surface-level Matching**: It only recognizes exact matches without understanding meaning.\n",
        "  \n",
        "  **Analogy**: Judging two chefs solely on whether they used identical ingredients, not on whether their dishes taste the same. \"Quick brown dog\" and \"fast auburn canine\" would be considered completely different.\n",
        "\n",
        "- **Word Order Sensitivity**: SACREBLEU can be overly sensitive to word order changes.\n",
        "  \n",
        "  **Analogy**: Considering \"I love pasta with tomato sauce\" and \"I love tomato sauce with pasta\" as significantly different statements.\n",
        "\n",
        "### BERTScore: The Semantic Understanding Approach\n",
        "\n",
        "BERTScore evaluates translations based on semantic similarity rather than exact word matches.\n",
        "\n",
        "#### How BERTScore Works (with Analogies)\n",
        "\n",
        "1. **Contextual Embeddings**: BERTScore converts words into vector representations that capture their meaning in context.\n",
        "   \n",
        "   **Analogy**: Understanding that \"automobile\" and \"car\" refer to the same thing, or that \"bank\" means something different in \"river bank\" versus \"money in the bank.\"\n",
        "\n",
        "2. **Token Matching**: It matches words based on semantic similarity.\n",
        "   \n",
        "   **Analogy**: Matching ingredients based on their culinary function rather than exact names - recognizing that \"cayenne\" could be a suitable substitute for \"chili powder\" in recipes.\n",
        "\n",
        "3. **Precision, Recall, and F1**: BERTScore calculates these metrics to produce a final score.\n",
        "   \n",
        "   **Analogy**: Evaluating a conversation by checking: did you cover all important points (recall)? Did you avoid irrelevant things (precision)? And how well did you balance both (F1 score)?\n",
        "\n",
        "#### Advantages\n",
        "\n",
        "- **Semantic Understanding**: BERTScore recognizes synonyms and paraphrases.\n",
        "  \n",
        "  **Analogy**: A music critic who can recognize when two different arrangements convey the same melody, even if using different instruments.\n",
        "\n",
        "- **Contextual Awareness**: It understands how words' meanings change based on context.\n",
        "  \n",
        "  **Analogy**: Understanding that \"cool\" means something different in \"the weather is cool\" versus \"that's a cool gadget.\"\n",
        "\n",
        "### Practical Example\n",
        "\n",
        "Reference: \"The cat quickly ran across the park.\"  \n",
        "Translation 1: \"The feline rapidly crossed the park.\"  \n",
        "Translation 2: \"A cat park the across quickly ran.\"\n",
        "\n",
        "**SACREBLEU Evaluation**:\n",
        "- Translation 1 would get a low score because few exact words match\n",
        "- Translation 2 might get a similar score because it has the same words, despite being nonsensical\n",
        "\n",
        "**BERTScore Evaluation**:\n",
        "- Translation 1 would get a high score because semantically it's very similar\n",
        "- Translation 2 would get a low score because semantically it makes no sense\n",
        "\n",
        "This is why using both metrics gives us a more complete picture of translation quality."
    ]
}

# Find the cell that contains the BLEU score result
bleu_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if 'source' in cell and isinstance(cell['source'], list):
        source_text = ''.join(cell['source'])
        if 'print("BLEU Score:"' in source_text:
            bleu_cell_index = i
            break

# Insert the markdown cell after the BLEU score cell
if bleu_cell_index is not None:
    notebook['cells'].insert(bleu_cell_index + 1, new_markdown_cell)
    print(f"Inserted explanation after cell {bleu_cell_index}")
else:
    print("Could not find the BLEU score cell")

# Save the updated notebook
with open('nmt_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!") 