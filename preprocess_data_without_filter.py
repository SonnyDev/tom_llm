#!/usr/bin/env python3
"""
Script to generate preprocessed.txt from the raw scores_*.csv files
This version KEEPS FC and TC trial states
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

    # NOTE: This version does NOT filter out FC and TC trial states

    gpt_data = []
    for model in ['GPT-3.5', 'GPT-4']:
        df_model = df_gpt[df_gpt['model'] == model].copy()
        score_cols = [col for col in df_model.columns if col.startswith('score')]

        for _, row in df_model.iterrows():
            task = task_name(row['task'])
            source = 'Original' if row['source'] == 'old' else 'New'

            # Calculate mean score across all score columns for this item
            scores = [row[col] for col in score_cols if pd.notna(row[col]) and row[col] != '']
            if scores:
                mean_score = np.mean([float(s) for s in scores])

                # Add one row per score column to match the expected format
                for i, score_col in enumerate(score_cols, 1):
                    score_val = row[score_col]
                    if pd.notna(score_val) and score_val != '':
                        gpt_data.append({
                            'task': task,
                            'trial': f'score{i}',
                            'source': source,
                            'model': model,
                            'score': float(score_val)
                        })

    df_gpt_processed = pd.DataFrame(gpt_data)

    # Process LLaMA data
    print("Processing LLaMA data...")
    df_llama = pd.read_csv('data/scores_llama.csv')

    llama_data = []
    score_cols_70b = [col for col in df_llama.columns if col.startswith('score_70B')]

    for _, row in df_llama.iterrows():
        task = task_name(row['task'])
        source = 'Original' if row['source'] == 'old' else 'New'

        for i, score_col in enumerate(score_cols_70b, 1):
            score_val = row[score_col]
            if pd.notna(score_val) and score_val != '':
                llama_data.append({
                    'task': task,
                    'trial': f'score{i}',
                    'source': source,
                    'model': 'LLaMA2-70B',
                    'score': float(score_val)
                })

    df_llama_processed = pd.DataFrame(llama_data)

    # Process Human data
    print("Processing Human data...")
    df_human = pd.read_csv('data/scores_human.csv')

    # NOTE: This version does NOT filter out FC and TC trial states

    human_data = []
    score_cols = [col for col in df_human.columns if col.startswith('score')]

    for _, row in df_human.iterrows():
        task = task_name(row['task'])
        source = 'Original' if row['source'] == 'old' else 'New'

        for i, score_col in enumerate(score_cols, 1):
            score_val = row[score_col]
            if pd.notna(score_val) and score_val != '' and score_val != 'NA':
                human_data.append({
                    'task': task,
                    'trial': f'score{i}',
                    'source': source,
                    'model': 'Human',
                    'score': float(score_val)
                })

    df_human_processed = pd.DataFrame(human_data)

    # Combine all data (including Human data)
    print("Combining all data...")
    df_all = pd.concat([df_gpt_processed, df_llama_processed, df_human_processed], ignore_index=True)

    # Group by task, trial, source, model and take mean
    print("Aggregating scores...")
    df_aggregated = df_all.groupby(['task', 'trial', 'source', 'model'], as_index=False)['score'].mean()

    # Define custom sort orders
    task_order = ['False Belief', 'Faux Pas', 'Hinting', 'Irony', 'Strange Stories']
    source_order = ['Original', 'New']
    model_order = ['GPT-3.5', 'GPT-4', 'Human', 'LLaMA2-70B']

    # Create categorical types with custom order
    df_aggregated['task'] = pd.Categorical(df_aggregated['task'], categories=task_order, ordered=True)
    df_aggregated['source'] = pd.Categorical(df_aggregated['source'], categories=source_order, ordered=True)
    df_aggregated['model'] = pd.Categorical(df_aggregated['model'], categories=model_order, ordered=True)

    # Extract numeric part from trial for proper sorting (score1, score2, ..., score10, etc.)
    df_aggregated['trial_num'] = df_aggregated['trial'].str.extract(r'(\d+)').astype(int)

    # Sort to match expected order: task, trial_num, source, model
    df_aggregated = df_aggregated.sort_values(['task', 'trial_num', 'source', 'model']).reset_index(drop=True)

    # Drop the temporary trial_num column
    df_aggregated = df_aggregated.drop('trial_num', axis=1)

    # Save to CSV with row index and quotes like the original
    print("Saving to preprocessed_with_fc_tc.txt...")
    df_aggregated.index = df_aggregated.index + 1
    df_aggregated.to_csv('data/preprocessed_with_fc_tc.txt', index=True, quoting=1)

    print(f"Done! Generated {len(df_aggregated)} rows")
    print(f"\nTasks: {sorted(df_aggregated['task'].unique())}")
    print(f"Models: {sorted(df_aggregated['model'].unique())}")
    print(f"Sources: {sorted(df_aggregated['source'].unique())}")

if __name__ == '__main__':
    main()
