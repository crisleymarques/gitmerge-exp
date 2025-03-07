from .conflict_resolution_generator import process_dataframe
from .client.gemini_client import generate_content

import pandas as pd


df = pd.read_json("data/eclipse_val_conflicts.jsonl", lines=True, nrows=3)

start_index = 0
max_requests = 1500 

result_df, last_processed_index = process_dataframe(df, 
                                                    'gemini-2.0-flash',
                                                    # "gemini-2.0-flash-thinking-exp-01-21",
                                                    generate_content, 
                                                    start_index, 
                                                    max_requests)
print("last_processed_index", last_processed_index)
result_df.to_json("output/solved_conflicts.json", orient="records")