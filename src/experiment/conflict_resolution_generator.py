from .utils import add_row_to_dataframe, handle_error
from .prompt import create_prompt
from typing import Tuple

import time
import pandas as pd


def process_dataframe(df, generate_content_fn, start_index=0, max_requests=None, wait_time=20) -> Tuple[pd.DataFrame, int]:
    """
    Generates conflict resolutions for a DataFrame of conflict information.

    Args:
        df: DataFrame with columns 'commit_sha', 'conflict_tuple', 'commit_message'
        generate_content_fn: Function to generate resolution content from a prompt
        start_index: Starting index for processing (default: 0)
        max_requests: Max number of rows to process (default: all remaining)
        wait_time: Seconds to wait between requests (default: 20)

    Returns:
        (DataFrame with resolutions, ending index)
    """
    res_df = pd.DataFrame(columns=['commit_sha', 'conflict_resolution'])
    total_rows = len(df)
    
    if max_requests is None:
        max_requests = total_rows - start_index
    
    end_index = min(start_index + max_requests, total_rows)
    
    for index in range(start_index, end_index):
        row = df.iloc[index]
        try:
            prompt = create_prompt(row['conflict_tuple'], row['commit_message'])
            
            response_text = generate_content_fn(prompt)
            print(f"Processed index {index}: {response_text}")
            
            res_df = add_row_to_dataframe(res_df, row['commit_sha'], response_text, index, total_rows)
            
        except Exception as e:
            res_df = handle_error(res_df, row, e, index)
        
        time.sleep(wait_time)
    
    return res_df, end_index

