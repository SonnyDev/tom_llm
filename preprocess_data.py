#!/usr/bin/env python3
"""
Script to generate preprocessed.txt from the raw scores_*.csv files
This version replicates the R code logic from Supplement Analysis Figures.Rmd lines 201-205
"""

import pandas as pd
import numpy as np

def task_name(task_name):
    """Simplify task names to match expected format"""
    if 'False Belief' in task_name:
        return 'False Belief'
    elif 'Irony' in task_name:
        return 'Irony'
    elif 'Strange Stories' in task_name:
        return 'Strange Stories'
    elif 'Faux Pas' in task_name:
        return 'Faux Pas'
    elif 'Hinting' in task_name:
        return 'Hinting'
    else:
        return task_name

def main():
    # Process GPT data
    print("Processing GPT data...")
    df_gpt = pd.read_csv('data/scores_gpt.csv')

    # Filter out FC and TC trial states (not part of the study)
    df_gpt = df_gpt[~df_gpt['trial_state'].isin(['FC', 'TC'])].copy()

    # Simplify task names
    df_gpt['task'] = df_gpt['task'].apply(task_name)

    # Map source: old -> Original, new -> New
    df_gpt['source'] = df_gpt['source'].map({'old': 'Original', 'new': 'New'})

    # Gather: transform score columns into rows (like R's gather function)
    score_cols = [col for col in df_gpt.columns if col.startswith('score')]
    df_gpt_long = df_gpt.melt(
        id_vars=['task', 'item', 'source', 'trial_state', 'model'],
        value_vars=score_cols,
        var_name='trial',
        value_name='score'
    )

    # Filter out NA values and convert score to numeric
    df_gpt_long = df_gpt_long[df_gpt_long['score'].notna()].copy()
    df_gpt_long = df_gpt_long[df_gpt_long['score'] != ''].copy()
    df_gpt_long['score'] = pd.to_numeric(df_gpt_long['score'], errors='coerce')
    df_gpt_long = df_gpt_long[df_gpt_long['score'].notna()].copy()

    # Rename LLaMA-70B to LLaMA2-70B to match reference
    df_gpt_long['model'] = df_gpt_long['model'].replace('LLaMA-70B', 'LLaMA2-70B')

    # Group by task, trial, source, model and calculate mean (averaging across all items)
    df_gpt_aggregated = df_gpt_long.groupby(['task', 'trial', 'source', 'model'], as_index=False)['score'].mean()

    print(f"GPT+LLaMA data: {len(df_gpt_aggregated)} rows")

    # Process Human data
    print("Processing Human data...")
    df_human = pd.read_csv('data/scores_human.csv')

    # Filter out FC and TC trial states
    df_human = df_human[~df_human['trial_state'].isin(['FC', 'TC'])].copy()

    # Simplify task names
    df_human['task'] = df_human['task'].apply(task_name)

    # Map source
    df_human['source'] = df_human['source'].map({'old': 'Original', 'new': 'New'})

    # Gather
    score_cols = [col for col in df_human.columns if col.startswith('score')]
    df_human_long = df_human.melt(
        id_vars=['task', 'item', 'source', 'trial_state'],
        value_vars=score_cols,
        var_name='trial',
        value_name='score'
    )

    # Add model column
    df_human_long['model'] = 'Human'

    # Filter out NA values
    df_human_long = df_human_long[df_human_long['score'].notna()].copy()
    df_human_long = df_human_long[df_human_long['score'] != ''].copy()
    df_human_long = df_human_long[df_human_long['score'] != 'NA'].copy()
    df_human_long['score'] = pd.to_numeric(df_human_long['score'], errors='coerce')
    df_human_long = df_human_long[df_human_long['score'].notna()].copy()

    # Group by task, trial, source, model and calculate mean
    df_human_aggregated = df_human_long.groupby(['task', 'trial', 'source', 'model'], as_index=False)['score'].mean()

    print(f"Human data: {len(df_human_aggregated)} rows")

    # Combine all data
    print("Combining all data...")
    df_all = pd.concat([df_gpt_aggregated, df_human_aggregated], ignore_index=True)

    # Define custom sort orders to match R output
    task_order = ['False Belief', 'Faux Pas', 'Hinting', 'Irony', 'Strange Stories']
    source_order = ['Original', 'New']
    model_order = ['GPT-3.5', 'GPT-4', 'Human', 'LLaMA2-70B']

    # Create categorical types with custom order
    df_all['task'] = pd.Categorical(df_all['task'], categories=task_order, ordered=True)
    df_all['source'] = pd.Categorical(df_all['source'], categories=source_order, ordered=True)
    df_all['model'] = pd.Categorical(df_all['model'], categories=model_order, ordered=True)

    # Extract numeric part from trial for proper sorting (score1, score2, ..., score10, etc.)
    df_all['trial_num'] = df_all['trial'].str.extract(r'(\d+)').astype(int)

    # Sort to match expected order: task, trial_num, source, model
    df_all = df_all.sort_values(['task', 'trial_num', 'source', 'model']).reset_index(drop=True)

    # Drop the temporary trial_num column
    df_all = df_all.drop('trial_num', axis=1)

    # Save to CSV with row index and quotes like the original
    print("Saving to preprocessed.txt...")
    df_all.index = df_all.index + 1
    df_all.to_csv('data/preprocessed.txt', index=True, quoting=1)

    print(f"Done! Generated {len(df_all)} rows")
    print(f"\nTasks: {df_all['task'].unique().tolist()}")
    print(f"Models: {df_all['model'].unique().tolist()}")
    print(f"Sources: {df_all['source'].unique().tolist()}")
    print(f"\nFirst few rows:")
    print(df_all.head(10))

if __name__ == '__main__':
    main()
