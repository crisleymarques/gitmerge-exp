from .utils import add_row_to_dataframe, handle_error
from .prompt import create_prompt

import time
import pandas as pd


def process_dataframe(df, model_name, generate_content_fn, start_index=0, max_requests=None, wait_time=20):
    res_df = pd.DataFrame(columns=['commit_sha', 'conflict_resolution'])
    total_rows = len(df)
    
    if max_requests is None:
        max_requests = total_rows - start_index
    
    end_index = min(start_index + max_requests, total_rows)
    
    for index in range(start_index, end_index):
        row = df.iloc[index]
        try:
            prompt = create_prompt(row['conflict_tuple'], row['commit_message'])
            
            response = generate_content_fn(model_name, prompt)
            
            response_text = response.text if hasattr(response, 'text') else response
            print(f"Processed index {index}: {response_text}")
            
            res_df = add_row_to_dataframe(res_df, row['commit_sha'], response_text, index, total_rows)
            
        except Exception as e:
            res_df = handle_error(res_df, row, e, index)
        
        time.sleep(wait_time)
    
    return res_df, end_index

