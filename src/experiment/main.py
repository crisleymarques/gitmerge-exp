from .conflict_resolution_generator import process_dataframe
from .config.cli_config import setup_cli_parser, get_llm_config_from_args
from .llm.llm_client import LLMClient
from .utils import get_project_root
import pandas as pd
import os


def main():
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    llm_config = get_llm_config_from_args(args)
    llm_client = LLMClient(llm_config)
    print(f"Using provider: {llm_config.provider}, model: {llm_config.model}")

    project_root = get_project_root()
    
    input_path = os.path.join(project_root, "data", "eclipse_val_conflicts.jsonl")
    df = pd.read_json(input_path, lines=True, nrows=1)
    
    start_index = 0
    max_requests = 1500 

    result_df, last_processed_index = process_dataframe(df,
                                                        llm_client.generate_content, 
                                                        start_index, 
                                                        max_requests)
    print("last_processed_index", last_processed_index)

    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "solved_conflicts.json")
    result_df.to_json(output_path, orient="records")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()