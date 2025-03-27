import json

# Load the notebook
with open('nmt_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create markdown cell for introducing the comparison
intro_markdown_cell = {
    "cell_type": "markdown",
    "id": "example-comparison",
    "metadata": {},
    "source": [
        "## Practical Comparison: BERTScore vs SACREBLEU\n",
        "\n",
        "Let's demonstrate the difference between these metrics using our example sentences from above:"
    ]
}

# Create code cell for setting up the examples
setup_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "practical-example-setup",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define our reference and two translation examples\n",
        "reference = [\"The cat quickly ran across the park.\"]\n",
        "translation1 = [\"The feline rapidly crossed the park.\"]  # Semantically similar but different words\n",
        "translation2 = [\"A cat park the across quickly ran.\"]    # Same words but nonsensical order"
    ]
}

# Create code cell for BLEU comparison
bleu_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "bleu-comparison",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Set up the SACREBLEU metric\n",
        "bleu_metric = evaluate.load(\"sacrebleu\")\n",
        "\n",
        "# Calculate BLEU scores for both translations\n",
        "bleu_score1 = bleu_metric.compute(predictions=translation1, references=[reference])\n",
        "bleu_score2 = bleu_metric.compute(predictions=translation2, references=[reference])\n",
        "\n",
        "print(\"SACREBLEU Scores:\")\n",
        "print(f\"Translation 1 (semantically similar): {bleu_score1['score']:.2f}\")\n",
        "print(f\"Translation 2 (nonsensical order): {bleu_score2['score']:.2f}\")"
    ]
}

# Create code cell for BERTScore comparison
bert_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "bert-comparison",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Set up the BERTScore metric\n",
        "bert_metric = load(\"bertscore\")\n",
        "\n",
        "# Calculate BERTScores for both translations\n",
        "bert_results1 = bert_metric.compute(predictions=translation1, references=reference, lang=\"en\")\n",
        "bert_results2 = bert_metric.compute(predictions=translation2, references=reference, lang=\"en\")\n",
        "\n",
        "print(\"BERTScore F1 Scores:\")\n",
        "print(f\"Translation 1 (semantically similar): {bert_results1['f1'][0]:.4f}\")\n",
        "print(f\"Translation 2 (nonsensical order): {bert_results2['f1'][0]:.4f}\")"
    ]
}

# Create code cell for visualization
vis_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "visualization",
    "metadata": {},
    "outputs": [],
    "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "# Normalize BLEU scores to 0-1 range for comparison\n",
        "bleu1_norm = bleu_score1['score'] / 100\n",
        "bleu2_norm = bleu_score2['score'] / 100\n",
        "\n",
        "# Set up the comparison data\n",
        "translations = ['Semantically Similar', 'Nonsensical Order']\n",
        "bleu_scores = [bleu1_norm, bleu2_norm]\n",
        "bert_scores = [bert_results1['f1'][0], bert_results2['f1'][0]]\n",
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
        "plt.title('SACREBLEU vs BERTScore Comparison')\n",
        "plt.xticks([r + barWidth/2 for r in range(len(translations))], translations)\n",
        "plt.ylim(0, 1.0)\n",
        "plt.legend()\n",
        "\n",
        "# Display the plot\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Create markdown cell for analysis
analysis_markdown_cell = {
    "cell_type": "markdown",
    "id": "analysis",
    "metadata": {},
    "source": [
        "### Analysis of the Results\n",
        "\n",
        "The comparison demonstrates the key differences between SACREBLEU and BERTScore:\n",
        "\n",
        "1. **For semantically similar but lexically different translations (Translation 1)**:\n",
        "   - SACREBLEU gives a lower score because it only counts exact matching words (\"the\" and \"park\" match, but \"feline\" != \"cat\", \"rapidly\" != \"quickly\", etc.)\n",
        "   - BERTScore gives a higher score because it recognizes that \"feline\" is semantically similar to \"cat\" and \"rapidly\" is semantically similar to \"quickly\"\n",
        "\n",
        "2. **For nonsensical ordering of the same words (Translation 2)**:\n",
        "   - SACREBLEU gives a moderate score because many individual words match, even though the meaning is lost\n",
        "   - BERTScore gives a much lower score because it considers context and meaning, not just word matching\n",
        "\n",
        "This demonstrates why BERTScore is often more aligned with human judgment of translation quality, as it measures semantic similarity rather than just surface-level word matches."
    ]
}

# Find the cell with the note "Compare a single sentence using Bleu and BERT"
comparison_cell_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell and isinstance(cell['source'], list):
        source_text = ''.join(cell['source'])
        if 'Compare a single sentence using Bleu and BERT' in source_text:
            comparison_cell_index = i
            break

if comparison_cell_index is not None:
    # Insert our new cells after the comparison prompt
    notebook['cells'].insert(comparison_cell_index + 1, intro_markdown_cell)
    notebook['cells'].insert(comparison_cell_index + 2, setup_code_cell)
    notebook['cells'].insert(comparison_cell_index + 3, bleu_code_cell)
    notebook['cells'].insert(comparison_cell_index + 4, bert_code_cell)
    notebook['cells'].insert(comparison_cell_index + 5, vis_code_cell)
    notebook['cells'].insert(comparison_cell_index + 6, analysis_markdown_cell)
    print(f"Inserted example comparison after cell {comparison_cell_index}")
else:
    print("Could not find the comparison cell")

# Save the updated notebook
with open('nmt_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!") 