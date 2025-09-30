import json
import sys
import os

def round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v) for v in obj]
    else:
        return obj

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} input.jsonl")
    sys.exit(1)

input_path = sys.argv[1]
tmp_path = input_path + '.tmp'

with open(input_path, 'r') as fin, open(tmp_path, 'w') as fout:
    for line in fin:
        obj = json.loads(line)
        obj = round_floats(obj)
        fout.write(json.dumps(obj) + '\n')

os.replace(tmp_path, input_path)