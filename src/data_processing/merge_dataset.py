from datetime import datetime

import argparse
import json
import os


def merge_datasets(dataset1_path, dataset2_path, output_path):
    """
    Merge two datasets containing solved conflicts.
    The dataset with the smaller ID will be placed first in the results array.
    Merge is only performed if both datasets use the same model.
    Only one metadata object is kept.
    """
    # Load datasets
    try:
        with open(dataset1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        
        with open(dataset2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in one of the input files")
        return
    
    # Check if models are the same
    if data1["metadata"]["model"] != data2["metadata"]["model"]:
        print(f"Error: Models are different - {data1['metadata']['model']} vs {data2['metadata']['model']}")
        return
    
    # Determine which dataset has the smaller IDs
    min_id1 = min(result["id"] for result in data1["results"]) if data1["results"] else float('inf')
    min_id2 = min(result["id"] for result in data2["results"]) if data2["results"] else float('inf')
    
    # Arrange datasets in order (smaller IDs first)
    first_dataset = data1 if min_id1 <= min_id2 else data2
    second_dataset = data2 if min_id1 <= min_id2 else data1
    
    merged_data = {
        "metadata": {
            "provider": first_dataset["metadata"]["provider"],
            "model": first_dataset["metadata"]["model"],
            "timestamp": datetime.now().isoformat(),
            "total_records": len(first_dataset["results"]) + len(second_dataset["results"]),
            "is_checkpoint": False
        },
        "results": first_dataset["results"] + second_dataset["results"]
    }
    
    # Keep additional metadata fields if they exist
    if "last_processed_index" in first_dataset["metadata"]:
        merged_data["metadata"]["last_processed_index"] = max(
            first_dataset["metadata"].get("last_processed_index", 0),
            second_dataset["metadata"].get("last_processed_index", 0)
        )
    
    # Save merged dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged dataset saved to {output_path}")
    print(f"Total records: {merged_data['metadata']['total_records']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two solved conflicts datasets')
    parser.add_argument('dataset1', help='Path to the first dataset JSON file')
    parser.add_argument('dataset2', help='Path to the second dataset JSON file')
    parser.add_argument('--output', '-o', default='data/output/merged_dataset.json',
                       help='Path to save the merged dataset (default: data/output/merged_dataset.json)')
    
    args = parser.parse_args()
    merge_datasets(args.dataset1, args.dataset2, args.output)
