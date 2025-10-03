# ToM LLM - Theory of Mind Analysis

This repository contains Python scripts and Jupyter notebooks for analyzing Theory of Mind (ToM) capabilities in Large Language Models (LLMs).

## Objective

This project aims to evaluate and compare the Theory of Mind capabilities of different language models (GPT-3.5, GPT-4, LLaMA2-70B) with human performance across multiple tasks:
- False Belief
- Faux Pas
- Hinting
- Irony
- Strange Stories

## Project Structure

- `preprocess_data.py`: Main script to preprocess raw data and generate `preprocessed.txt`
- `preprocess_data_without_filter.py`: Alternative version without FC/TC filtering
- `generate_figure2.ipynb`: Notebook to generate figure 2
- `generate_figures.ipynb`: Notebook to generate analysis figures
- `data/`: Folder containing raw data (`scores_gpt.csv`, `scores_human.csv`) and preprocessed data
- `figures/`: Folder for generated figures

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess the data

```bash
python preprocess_data.py
```

This script:
- Loads raw data from `data/scores_gpt.csv` and `data/scores_human.csv`
- Filters out FC and TC trial states
- Aggregates scores by task, source, and model
- Generates `data/preprocessed.txt`

### 2. Generate figures

Open and run the Jupyter notebooks:

```bash
jupyter notebook generate_figures.ipynb
# or
jupyter notebook generate_figure2.ipynb
```

## Required Data

Ensure the `data/` folder contains:
- `scores_gpt.csv`: LLM model scores
- `scores_human.csv`: Human participant scores
