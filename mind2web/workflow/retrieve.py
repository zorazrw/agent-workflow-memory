"""Retrieve workflows given a query."""

import os
import json
import random
import argparse

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# %% load test examples
def get_examples(data_dir: str, website: str = None) -> list[dict]:
    print("Start loading data files...")
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    examples = []
    for p in paths:
        data = json.load(open(p, 'r'))
        if website is not None:
            data = [ex for ex in data if website==ex["website"]]
        examples.extend(data)
    print(f"Collected {len(examples)} examples about website {website}")
    return examples


# %% load workflows
def clean_workflow_name(name: str) -> str:
    if ':' in name: 
        name = name[name.index(':')+1:].strip()
    if '`' in name:
        s = name.index('`')
        name = name[s+1: ]
        if '`' in name:
            e = name.index('`')
            name = name[: e]
    return f"## {name}"

def load_workflows(path: str) -> list[dict]:
    """Load workflow blocks in the given file path."""
    website = path.split('/')[-1].split('.')[0].split('_')[0]
    blocks = open(path, 'r').read().split('\n\n')

    def check_workflow(text: str) -> dict | None:
        lines = text.strip().split('\n')
        if len(lines) < 4: return None
        name = clean_workflow_name(lines[0].lstrip('#').strip())
        docstr = lines[1].strip()
        return {
            "website": website,
            "name": name, "docstring": docstr, 
            "content": '\n'.join([name] + lines[1: ])
        }
    
    workflows = [check_workflow(b) for b in blocks]
    workflows = [w for w in workflows if w is not None]
    return workflows

# %% retrieve workflows
def build_memory(workflows: list[dict], memory_path: str):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    metadatas = [{"name": i} for i in range(len(workflows))]
    texts = [f"{w['name']}\n{w['docstring']}" for w in workflows]
    memory = FAISS.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
    )
    if memory_path is not None:
        memory.save_local(memory_path)
    return metadatas
    

def get_ids_and_scores(memory, query: str, top_k: int) -> tuple[list[str], list[float]]:
    docs_and_similarities = memory.similarity_search_with_score(query, top_k)
    retrieved_ids, scores = [], []
    for doc, score in docs_and_similarities:
        retrieved_ids.append(doc.metadata["name"])
        scores.append(score)
    return retrieved_ids, scores


# %% main pipeline
def main():
    # enumerate workflow files
    suffix = ".txt" if args.workflow_suffix is None else f"_{args.suffix}.txt"
    workflow_files = []
    for (dirpath, dirnames, filenames) in os.walk(args.workflow_dir):
        workflow_files.extend([
            os.path.join(dirpath, f) 
            for f in filenames if f.endswith(suffix)
        ])
    print(f"Collected #{len(workflow_files)} workflow files in total.")

    # parse workflows from each file
    workflows = []
    for wf in workflow_files:
        workflows.extend(load_workflows(wf))
    print(f"Collected #{len(workflows)} from files.")

    # select workflows
    if args.mode == "random":
        selected_workflows = random.sample(workflows, args.top_k)
    elif args.mode == "semantic":
        memory = build_memory(workflows, args.memory_path)
        examples = get_examples(args.data_dir, args.website)
        queries = [ex["confirmed_task"] for ex in examples]
        retrieved_ids_and_scores = []
        for ex in examples:
            rids, rscores = get_ids_and_scores(memory, query=ex["confirmed_task"], top_k=args.top_k)
            retrieved_ids_and_scores.extend([(rid,rscr) for rid,rscr in zip(rids, rscores)])
        retrieved_ids_and_scores = sorted(retrieved_ids_and_scores, key=lambda x:-x[1])
        selected_workflows = [workflows[i] for i,s in retrieved_ids_and_scores[:args.top_k]]
    else:
        raise ValueError
    
    # write selected workflows to the output path
    with open(args.output_path, 'w') as fw:
        fw.write('\n\n'.join([w["content"] for w in selected_workflows]))


def ablation(): # retrieve from training examples
    train_examples = json.load(open("../data/memory/exemplars.json", 'r'))
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    metadatas = [{"name": i} for i in range(len(train_examples))]
    texts = [ex[0]["specifier"] for ex in train_examples]
    memory = FAISS.from_texts(texts=texts, embedding=embedding, metadatas=metadatas)

    # get test examples
    examples = get_examples(args.data_dir, args.website)
    retrieved_ids_and_scores = []
    for ex in examples:
        rids, rscores = get_ids_and_scores(memory, query=ex["confirmed_task"], top_k=args.top_k)
        retrieved_ids_and_scores.extend([(rid,rscr) for rid,rscr in zip(rids, rscores)])
    print("Top Retrieved Item: ",retrieved_ids_and_scores[0])
    retrieved_ids_and_scores = sorted(retrieved_ids_and_scores, key=lambda x:-x[1])
    selected_examples = [train_examples[i] for i,s in retrieved_ids_and_scores[:args.top_k]]
    
    # write selected examples to the output path
    selected_examples = [
        '\n'.join([item["content"] for item in sex]) 
        for sex in selected_examples
    ]
    with open(args.output_path, 'w') as fw:
        fw.write('\n\n'.join(selected_examples))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # offline workflows
    parser.add_argument("--workflow_dir", type=str, default=None,
                        help="Directory of workflows to retrieve from.")
    parser.add_argument("--workflow_suffix", type=str, default=None,
                        help="Specified suffix of workflow files to load.")
    
    # test data
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--website", type=str, default=None)

    # retrieval
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "semantic"])
    parser.add_argument("--memory_path", type=str, default="memory")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top-relevant workflows to save.")

    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output the collected (relevant) workflows.")
    # ablation
    parser.add_argument("--run_ablation", action="store_true",
                        help="If run ablation study to retrieve entire examples.")

    args = parser.parse_args()
    
    if args.mode == "semantic":
        assert (args.data_dir is not None) and (args.website is not None) and (args.workflow_dir is not None)
    
    if args.run_ablation: ablation()
    else: main()
