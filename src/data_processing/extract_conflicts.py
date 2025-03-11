import json

def extract_repositories(input_files, output_file, field_name, keyword):
    record_id = 1
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    data = json.loads(line)
                    if keyword.lower() in data.get(field_name, "").lower():
                        data['id'] = record_id
                        record_id += 1
                        json.dump(data, outfile)
                        outfile.write('\n')


input_files = [
    'data/dataset_chat_merge/dataset_val_conflict.jsonl'
]
extract_repositories(input_files, 'data/filtered_repositories.jsonl', 'repository_name', 'eclipse/')
