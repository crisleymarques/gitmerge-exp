import json
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
from pathlib import Path


dataset_path = 'data/elastic_train_conflicts.jsonl'
resolution_path = 'data/output/2-elastic-gemini_20250311/solved_conflicts_gemini-2.0-flash_20250311_071912.json'

def normalize_code(code):
    """
    Normalize code by removing code block markers, extra whitespace,
    and standardizing line endings.
    """
    # Remove code block markers (```java, ```, etc.)
    code = re.sub(r'```\w*\n?|\n?```', '', code)
    
    # Normalize whitespace (trim leading/trailing, standardize line endings)
    code = code.strip()
    
    # Remove extra whitespace at end of lines
    code = re.sub(r' +$', '', code, flags=re.MULTILINE)
    
    return code

def calculate_similarity(text1, text2):
    """
    Calculate similarity between two text strings using SequenceMatcher.
    Returns a score between 0 and 100.
    """
    if not text1 and not text2:
        return 100  # Both empty means they're identical
    if not text1 or not text2:
        return 0    # One empty means no similarity
    
    text1 = normalize_code(text1)
    text2 = normalize_code(text2)
    
    similarity = SequenceMatcher(None, text1, text2).ratio() * 100
    
    return similarity

def load_original_conflicts():
    """
    Load the original conflicts from the dataset file.
    """
    conflicts = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            conflicts[data['id']] = {
                'original_resolution': data['conflict_tuple']['resolution'],
                'a_content': data['conflict_tuple']['a_content'],
                'b_content': data['conflict_tuple']['b_content'],
                'base_content': data['conflict_tuple']['base_content'],
                'filename': data['filename'],
                'commit_sha': data['commit_sha']
            }
    return conflicts

def load_generated_resolutions():
    """
    Load the generated resolutions from the results file.
    """
    with open(resolution_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    resolutions = {}
    for item in results:
        resolutions[item['id']] = {
            'generated_resolution': item['conflict_resolution'],
            'commit_sha': item['commit_sha']
        }
    
    return resolutions

def evaluate_resolutions():
    """
    Compare original and generated resolutions and create evaluation metrics.
    """
    original_conflicts = load_original_conflicts()
    generated_resolutions = load_generated_resolutions()
    
    evaluation = []
    
    for conflict_id in original_conflicts:
        if conflict_id in generated_resolutions:
            original = original_conflicts[conflict_id]['original_resolution']
            generated = generated_resolutions[conflict_id]['generated_resolution']
            
            original_norm = normalize_code(original)
            generated_norm = normalize_code(generated)
            
            is_empty_resolution = original_norm == ""
            
            exact_match = original_norm == generated_norm
            similarity = calculate_similarity(original, generated)
            
            a_content = original_conflicts[conflict_id]['a_content']
            b_content = original_conflicts[conflict_id]['b_content']
            base_content = original_conflicts[conflict_id]['base_content']
            
            a_similarity = calculate_similarity(generated, a_content)
            b_similarity = calculate_similarity(generated, b_content)
            base_similarity = calculate_similarity(generated, base_content)
            
            # Simple heuristic to classify resolution approach
            approach = "custom"
            highest_sim = max(a_similarity, b_similarity, base_similarity)
            
            if highest_sim > 90:  # Threshold for considering it's the same
                if highest_sim == a_similarity:
                    approach = "chose_a"
                elif highest_sim == b_similarity:
                    approach = "chose_b"
                elif highest_sim == base_similarity:
                    approach = "chose_base"
            
            evaluation.append({
                'id': conflict_id,
                'exact_match': exact_match,
                'similarity': similarity,
                'is_empty_resolution': is_empty_resolution,
                'resolution_approach': approach,
                'a_similarity': a_similarity,
                'b_similarity': b_similarity,
                'base_similarity': base_similarity,
                'filename': original_conflicts[conflict_id]['filename'],
                'commit_sha': original_conflicts[conflict_id]['commit_sha']
            })
    
    df = pd.DataFrame(evaluation)
    return df

def json_serialize(obj):
    """
    Converte valores NumPy para tipos Python nativos para permitir serialização JSON.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj

def save_evaluation_results(df):
    """
    Save evaluation results to JSON files.
    """
    print("=== Evaluation Summary ===")
    print(f"Total conflicts evaluated: {len(df)}")
    print(f"Exact matches: {df['exact_match'].sum()} ({df['exact_match'].mean()*100:.2f}%)")
    print(f"Average similarity: {df['similarity'].mean():.2f}%")
    
    output_dir = Path('data/evaluation')
    output_dir.mkdir(exist_ok=True)
    
    # Ensure path is structured to include model name
    model_name = resolution_path.split('/')[-2].split('-')[1]  # Extract 'elastic' from path
    
    # Save main evaluation results
    results_filename = f"{model_name}-resolution_evaluation.json"
    df.to_json(output_dir / results_filename, orient='records', indent=2)
    
    # Save file extension summary
    df['file_extension'] = df['filename'].apply(lambda x: x.split('.')[-1] if '.' in x else 'unknown')
    ext_summary = df.groupby('file_extension').agg({
        'exact_match': ['count', 'mean'],
        'similarity': 'mean'
    })
    ext_summary.columns = ['count', 'exact_match_rate', 'avg_similarity']
    ext_summary = ext_summary.sort_values('count', ascending=False)
    
    # Convert the multi-index DataFrame to a format suitable for JSON
    ext_summary_json = ext_summary.reset_index().to_dict(orient='records')
    ext_summary_filename = f"{model_name}-file_extension_summary.json"
    with open(output_dir / ext_summary_filename, 'w') as f:
        json.dump(ext_summary_json, f, indent=2, default=json_serialize)
    
    print(f"Results saved to '{output_dir / results_filename}' and '{output_dir / ext_summary_filename}'")
    
    return results_filename, ext_summary_filename

def main():
    """
    Main function to run the evaluation.
    """
    print("Starting evaluation of generated conflict resolutions...")
    df = evaluate_resolutions()
    save_evaluation_results(df)
    print("Evaluation complete.")
    return df

if __name__ == "__main__":
    df = main()
