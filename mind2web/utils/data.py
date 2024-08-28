import os
import json
import pickle

# %% load data
def load_json(data_dir, folder_name):
    folder_path = os.path.join(data_dir, folder_name)
    print(f"Data path: {folder_path}")
    data_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".json")
    ]
    data_paths = sorted(data_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Construct trajectory dataset
    samples = []
    for data_path in data_paths:
        with open(data_path, "r") as f:
            samples.extend(json.load(f))
    print("# of samples:", len(samples))

    return samples


def add_scores(
    examples: list[dict], candidate_results: dict = None,
    score_path: str = "data/scores_all_data.pkl"
):
    """Add prediction scores and ranks to candidate elements."""
    if candidate_results is None:
        with open(score_path, "rb") as f:
            candidate_results = pickle.load(f)

    for sample in examples:
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_results["scores"][sample_id][candidate_id]
                    candidate["rank"] = candidate_results["ranks"][sample_id][candidate_id]
    
    return examples


# %% workflow induction
def format_examples(examples: list[dict], prefix: str = None, suffix: str = None) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(f"Query #{i+1}: {ex['confirmed_task']}")
        lines.append("Actions and Environments:")
        lines.extend(ex["action_reprs"])
        lines.append("")
    prompt = '\n'.join(lines)
    if prefix is not None:
        prompt = prefix + '\n' + prompt
    if suffix is not None:
        prompt += '\n\n' + suffix
    return prompt 



# %% model generation
def is_website_header(block: str, website: str) -> bool:
    lines = block.strip().split('\n')
    if len(lines) > 1: return False
    text = lines[0].strip()
    if text.startswith("#") and text.lower().endswith(website):
        return True
    return False

def filter_workflows(text: str, website: str) -> str:
    blocks = text.split('\n\n')
    for i,b in enumerate(blocks):
        if is_website_header(b, website):
            blocks = blocks[i+1: ]
            break
    
    for i,b in enumerate(blocks):
        if is_website_header(b, "delta"):
            blocks = blocks[: i]
            break

    blocks = [b for b in blocks if "delta" not in b.lower()]
    return '\n\n'.join(blocks)