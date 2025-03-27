# Load the notebook
import json

# Load the notebook
with open('nmt_notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create a new markdown cell that explains why someone would use SACREBLEU over BERTScore
sacrebleu_vs_bertscore_cell = {
    "cell_type": "markdown",
    "id": "sacrebleu-vs-bertscore",
    "metadata": {},
    "source": [
        "## Why Use SACREBLEU Despite BERTScore's Advantages?\n",
        "\n",
        "Despite BERTScore's advantages in semantic understanding, SACREBLEU remains valuable for several practical reasons:\n",
        "\n",
        "1. **Computational efficiency** - SACREBLEU is significantly faster and requires less computational resources than BERTScore, which uses large neural models.\n",
        "\n",
        "2. **Interpretability** - SACREBLEU's calculation is transparent and easier to understand, making it more accessible for debugging and explaining to stakeholders.\n",
        "\n",
        "3. **Established benchmark** - As the industry standard for years, SACREBLEU enables fair comparison with existing published results in academic literature.\n",
        "\n",
        "4. **Language coverage** - SACREBLEU works consistently across all languages, while BERTScore's performance can vary depending on the language representation in its underlying model.\n",
        "\n",
        "5. **Deterministic results** - SACREBLEU produces the same score every time for the same input, whereas neural-based methods can have slight variations.\n",
        "\n",
        "6. **Focus on fluency** - Its n-gram matching approach is particularly good at assessing grammatical correctness and fluency, which complements BERTScore's semantic focus.\n",
        "\n",
        "In practice, most researchers use both metrics together to get a more comprehensive evaluation of translation quality. SACREBLEU focuses on surface-level accuracy while BERTScore captures meaning preservation, giving complementary perspectives on translation quality."
    ]
}

# Find the appropriate location to insert the new cell
# We'll place it after the analysis cell but before the French example
analysis_cell_index = None
french_example_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'id' in cell and cell['id'] == 'analysis':
        analysis_cell_index = i
    if cell['cell_type'] == 'markdown' and 'id' in cell and cell['id'] == 'french-example':
        french_example_index = i
        break

# Insert the new markdown cell after the analysis cell
if analysis_cell_index is not None and french_example_index is not None:
    insert_index = analysis_cell_index + 1
    notebook['cells'].insert(insert_index, sacrebleu_vs_bertscore_cell)
    print(f"Inserted explanation after the analysis cell and before the French example")
else:
    # If we can't find both indexes, we'll look for just the analysis cell
    if analysis_cell_index is not None:
        insert_index = analysis_cell_index + 1
        notebook['cells'].insert(insert_index, sacrebleu_vs_bertscore_cell)
        print(f"Inserted explanation after the analysis cell")
    else:
        # If we can't find the analysis cell, we'll insert after the visualization cell
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'visualization':
                insert_index = i + 1
                notebook['cells'].insert(insert_index, sacrebleu_vs_bertscore_cell)
                print(f"Inserted explanation after the visualization cell")
                break
        else:
            # If we can't find any of the specific cells, add to the end
            notebook['cells'].append(sacrebleu_vs_bertscore_cell)
            print("Added explanation to the end of the notebook")

# Save the updated notebook
with open('nmt_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!") 