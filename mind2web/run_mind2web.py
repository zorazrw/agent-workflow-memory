import os
import argparse
from tqdm import tqdm
from memory import eval_sample
from utils.data import load_json, add_scores

import logging
logger = logging.getLogger("atm")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def main():
    examples = load_json(args.data_dir, args.benchmark)
    examples = [s for s in examples if s["website"] == args.website]
    print(f"Filtering down to #{len(examples)} examples on website [{args.website}]")
    examples = add_scores(examples) # add prediction scores and ranks to elements

    if args.end_idx is None:
        args.end_idx = len(examples)
    for i in tqdm(range(args.start_idx, args.end_idx)):
        if args.mode == "memory":
            eval_sample(i, args, examples[i])
        elif args.mode == "action":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported workflow format: {args.workflow_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--memory_path", type=str, default="data/memory")
    parser.add_argument("--log_dir", type=str, default="results")

    # model
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--temperature", type=float, default=0.0)

    # env context
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=1)

    # workflow
    parser.add_argument("--website", type=str, required=True)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--subdomain", type=str, default=None)
    parser.add_argument("--workflow_path", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="workflow")

    # ablation
    parser.add_argument("--mode", type=str, default="memory", choices=["memory", "action"])
    parser.add_argument("--start_idx", type=int, default=0, help="Select example index.")
    parser.add_argument("--end_idx", type=int, default=None, help="Select example index.")

    args = parser.parse_args()

    # sanity check
    if not os.path.exists(args.workflow_path): open(args.workflow_path, 'w').close()
    if args.retrieve_top_k != 1: print(f"Suggest set `retrieve_top_k` to 1, currently as {args.retrieve_top_k}")

    main()
