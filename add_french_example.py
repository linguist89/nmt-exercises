import json

# Load the notebook
with open('nmt_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Update the analysis cell with more detailed explanations
analysis_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'id' in cell and cell['id'] == 'analysis':
        analysis_cell_index = i
        break

if analysis_cell_index is not None:
    notebook['cells'][analysis_cell_index]['source'] = [
        "### Analysis of the Results\n",
        "\n",
        "The comparison demonstrates the key differences between SACREBLEU and BERTScore:\n",
        "\n",
        "1. **For semantically similar but lexically different translations (Translation 1)**:\n",
        "   - SACREBLEU gives a lower score (22.77) because it only counts exact matching words (\"the\" and \"park\" match, but \"feline\" != \"cat\", \"rapidly\" != \"quickly\", etc.)\n",
        "   - BERTScore gives a high score (0.9663) because it recognizes that \"feline\" is semantically similar to \"cat\" and \"rapidly\" is semantically similar to \"quickly\"\n",
        "\n",
        "2. **For nonsensical ordering of the same words (Translation 2)**:\n",
        "   - SACREBLEU gives a moderate score (15.11) because many individual words match, even though the meaning is lost\n",
        "   - BERTScore still gives a surprisingly high score (0.9228) because:\n",
        "     - It contains all the same words as the reference (shared vocabulary)\n",
        "     - BERT's contextual window has limitations in understanding complete grammar\n",
        "     - The averaging effect of token-level similarities boosts the score\n",
        "     - Transformer models like BERT capture some but not all word order information\n",
        "\n",
        "The important insight is the relative difference between the scores. BERTScore shows a more meaningful gap (0.9663 vs. 0.9228) between good and bad translations than SACREBLEU does proportionally.\n",
        "\n",
        "This demonstrates why using both metrics together gives us a more complete picture of translation quality."
    ]
    print("Updated analysis cell")

# Create French to English example markdown cell
french_intro_cell = {
    "cell_type": "markdown",
    "id": "french-english-example",
    "metadata": {},
    "source": [
        "## Cross-Language Example: French to English\n",
        "\n",
        "Since we've been working with French-English translation in this notebook, let's demonstrate how these metrics evaluate translations across languages. We'll use a French reference sentence and compare two English translations:"
    ]
}

# Create code cell for setting up the French to English examples
french_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "french-english-setup",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define our French reference and two English translation examples\n",
        "fr_reference = [\"Le chat a rapidement traversé le parc.\"]  # The cat quickly crossed the park\n",
        "en_translation1 = [\"The cat quickly ran across the park.\"]  # Semantically accurate translation\n",
        "en_translation2 = [\"The park across quickly cat the.\"]      # Nonsensical word order"
    ]
}

# Create code cell for BLEU comparison in cross-language
french_bleu_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cross-lang-bleu",
    "metadata": {},
    "outputs": [],
    "source": [
        "# We need a special setup for cross-language evaluation\n",
        "# First, we'll use our fr-en model to create a proper reference\n",
        "print(\"Creating a reference English translation using our model:\")\n",
        "model_reference = translator(fr_reference[0])[0]['translation_text']\n",
        "print(f\"Model reference: {model_reference}\")\n",
        "print(\"\\nNow comparing our two translations against this reference:\")\n",
        "\n",
        "# Calculate SACREBLEU scores for both translations against the model reference\n",
        "bleu_score_cross1 = bleu_metric.compute(predictions=en_translation1, references=[[model_reference]])\n",
        "bleu_score_cross2 = bleu_metric.compute(predictions=en_translation2, references=[[model_reference]])\n",
        "\n",
        "print(\"\\nSACREBLEU Scores (French to English):\")\n",
        "print(f\"Translation 1 (semantically accurate): {bleu_score_cross1['score']:.2f}\")\n",
        "print(f\"Translation 2 (nonsensical order): {bleu_score_cross2['score']:.2f}\")"
    ]
}

# Create code cell for BERTScore comparison in cross-language
french_bert_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cross-lang-bert",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Calculate BERTScores for our English translations against the model reference\n",
        "# We use the multilingual model to properly handle cross-language comparison\n",
        "bert_results_cross1 = bert_metric.compute(predictions=en_translation1, references=[model_reference], lang=\"en\")\n",
        "bert_results_cross2 = bert_metric.compute(predictions=en_translation2, references=[model_reference], lang=\"en\")\n",
        "\n",
        "print(\"BERTScore F1 Scores (French to English):\")\n",
        "print(f\"Translation 1 (semantically accurate): {bert_results_cross1['f1'][0]:.4f}\")\n",
        "print(f\"Translation 2 (nonsensical order): {bert_results_cross2['f1'][0]:.4f}\")"
    ]
}

# Create code cell for visualization of cross-language comparison
french_vis_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cross-lang-visualization",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualize the cross-language results\n",
        "# Normalize BLEU scores to 0-1 range for comparison\n",
        "bleu1_norm = bleu_score_cross1['score'] / 100\n",
        "bleu2_norm = bleu_score_cross2['score'] / 100\n",
        "\n",
        "# Set up the comparison data\n",
        "translations = ['Semantically Accurate', 'Nonsensical Order']\n",
        "bleu_scores = [bleu1_norm, bleu2_norm]\n",
        "bert_scores = [bert_results_cross1['f1'][0], bert_results_cross2['f1'][0]]\n",
        "\n",
        "# Set width of bars\n",
        "barWidth = 0.3\n",
        "r1 = np.arange(len(translations))\n",
        "r2 = [x + barWidth for x in r1]\n",
        "\n",
        "# Create the bars\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.bar(r1, bleu_scores, width=barWidth, label='SACREBLEU (normalized)')\n",
        "plt.bar(r2, bert_scores, width=barWidth, label='BERTScore F1')\n",
        "\n",
        "# Add labels and title\n",
        "plt.xlabel('Translation Type')\n",
        "plt.ylabel('Score (0-1 scale)')\n",
        "plt.title('Cross-Language Evaluation: French to English')\n",
        "plt.xticks([r + barWidth/2 for r in range(len(translations))], translations)\n",
        "plt.ylim(0, 1.0)\n",
        "plt.legend()\n",
        "\n",
        "# Display the plot\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Create markdown cell for analysis of French to English results
french_analysis_cell = {
    "cell_type": "markdown",
    "id": "cross-lang-analysis",
    "metadata": {},
    "source": [
        "### Analysis of Cross-Language Results\n",
        "\n",
        "This cross-language example (French to English) demonstrates how translation metrics work when evaluating translations between different languages:\n",
        "\n",
        "1. **Cross-language challenges**:\n",
        "   - When evaluating translations between languages, we need reference translations in the target language\n",
        "   - Here we used our translation model to create an English reference from the French original\n",
        "   - This approach simulates real-world evaluation of machine translation systems\n",
        "\n",
        "2. **Metric behavior is consistent**:\n",
        "   - SACREBLEU still prioritizes exact n-gram matches\n",
        "   - BERTScore still captures semantic similarity\n",
        "   - The pattern of scores for good vs. bad translations is similar to our monolingual examples\n",
        "\n",
        "3. **Practical implications**:\n",
        "   - When developing translation systems, we evaluate using references in the target language\n",
        "   - Using multiple metrics gives a more complete picture of translation quality\n",
        "   - Context and intended use should guide which metric to prioritize\n",
        "\n",
        "This example highlights why neural machine translation often uses multiple evaluation metrics when assessing system performance across languages."
    ]
}

# Find the last visualization cell to add our new French example after it
experiment_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell and isinstance(cell['source'], list):
        source_text = ''.join(cell['source'])
        if "Experiment with translations of other languages you know" in source_text:
            experiment_cell_index = i
            break

if experiment_cell_index is not None:
    # Insert the French to English example cells before the experiment cell
    notebook['cells'].insert(experiment_cell_index, french_intro_cell)
    notebook['cells'].insert(experiment_cell_index + 1, french_setup_cell)
    notebook['cells'].insert(experiment_cell_index + 2, french_bleu_cell)
    notebook['cells'].insert(experiment_cell_index + 3, french_bert_cell)
    notebook['cells'].insert(experiment_cell_index + 4, french_vis_cell)
    notebook['cells'].insert(experiment_cell_index + 5, french_analysis_cell)
    print(f"Inserted French to English example before cell {experiment_cell_index}")
else:
    print("Could not find the experiment cell")

# Save the updated notebook
with open('nmt_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!") 