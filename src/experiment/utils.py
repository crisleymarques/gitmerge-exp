import pandas as pd
import os

def add_row_to_dataframe(df, commit_sha, conflict_resolution, index, total_rows):
    new_row = pd.DataFrame([{
        'commit_sha': commit_sha,
        'conflict_resolution': conflict_resolution
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    print(f"Processed row {index + 1}/{total_rows}")
    print("-----------------------------------------------------------------")
    return df


def handle_error(df, row, error, index):
    print(f"Error processing row {index + 1}: {error}")
    new_row = pd.DataFrame([{
        'commit_sha': row['commit_sha'],
        'conflict_resolution': f"Erro ao gerar: {error}"
    }])

    return pd.concat([df, new_row], ignore_index=True)


def get_project_root():
    """Get the absolute path to the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))