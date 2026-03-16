import json
import os
from datasets import load_dataset

def prepare_dataset():
    # We use train + test splits as typical for RL if we want more data, 
    # as specified in benchmark_plan.md.
    ds = load_dataset("google-research-datasets/mbpp", split="train+test")
    
    out_path = os.path.join(os.path.dirname(__file__), "mbpp_rl.jsonl")
    with open(out_path, "w") as f:
        for row in ds:
            # "text" contains the natural language description
            prompt = row.get("text", "")
            test_list = row.get("test_list", [])
            # Package into the format SLIME's Dataset loader expects:
            # - input-key: "prompt"
            # - metadata-key: "metadata"
            item = {
                "prompt": prompt,
                "metadata": {
                    "test_list": test_list,
                    "task_id": row.get("task_id", -1)
                }
            }
            f.write(json.dumps(item) + "\n")
            
    print(f"Prepared {len(ds)} MBPP examples at {out_path}")

if __name__ == "__main__":
    prepare_dataset()
