# Algorithm for Entity Linking

This script enriches a CSV file with structured data from Wikipedia and Wikidata.

## What It Does

For each row of the input CSV:

1. The text in the **second column** is analyzed by a local LLM (Gemma2 9b) via Ollama (https://ollama.com/). Ollama is required.
2. The LLM extracts keywords and maps them to Wikipedia titles.
3. Wikipedia returns the corresponding **Wikidata QIDs**.
4. Wikidata (via SPARQL) provides:
   - English labels
   - Image URLs
   - Geographic coordinates (if available)

A new enriched CSV file is created.

---

## Input

- A CSV file  
- Header in the first row  
- Text to analyze in the **second column**  

Ollama must be installed and running locally.

---

## Output

The script generates: input.csv → input_qids.csv

---

New columns added:

- `wikidata_qids`
- `wikidata_names_en`
- `wikidata_images`
- `wikidata_coords`

Entities are separated by `,`  
Multiple images per entity are separated by `|`  
Missing values are written as `null`.

---

## Run

insert the CSV path in the INPUT_CSV_PATH variable, and run the main.py

