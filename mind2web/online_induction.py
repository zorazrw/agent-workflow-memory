"""Induce Workflows from Past Agent Experiences."""

import os
import json
import argparse
from utils.data import load_json, format_examples, filter_workflows

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI()


def is_io_dict(item: dict | str) -> bool:
    if isinstance(item, dict) and ("input" in item) and ("output" in item): return True
    return False

def get_trajectory(path: str):
    trajectory = []
    result = json.load(open(path, 'r'))
    for item in result:
        if not is_io_dict(item): continue
        step = {
            "env": "# " + item["input"][-1]["content"],
            "action": item["output"],
        }
        trajectory.append(step)
    return trajectory


def main():
    samples = load_json(args.data_dir, args.benchmark)
    print(f"Loaded #{len(samples)} test examples")
    samples = [s for s in samples if s["website"] == args.website]
    print(f"Filtering down to #{len(samples)} examples on website [{args.website}]")
    
    # load model predictions and format examples
    result_files = [os.path.join(args.results_dir, f) for f in os.listdir(args.results_dir)]
    result_list = [get_trajectory(rf) for rf in result_files]
    examples = []
    for r, s in zip(result_list, samples):
        examples.append({
            "confirmed_task": s["confirmed_task"],
            "action_reprs": [step["env"] + '\n' + step["action"] for step in r],
        })
    prompt = format_examples(examples, args.prefix, args.suffix)

    # transform to workflows
    INSTRUCTION = open(args.instruction_path, 'r').read()
    ONE_SHOT = open(args.one_shot_path, 'r').read()
    domain, subdomain, website = samples[0]["domain"], samples[0]["subdomain"], samples[0]["website"]
    prompt = '\n\n'.join([INSTRUCTION, ONE_SHOT, f"Website: {domain}, {subdomain}, {website}\n{prompt}"])
    response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
    ).choices[0].message.content
    response = filter_workflows(response, args.website)

    # save to file
    with open(args.output_path, 'w') as fw:
        fw.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--website", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # model
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=str, default=0.0)
    # prompt
    parser.add_argument("--instruction_path", type=str, default="prompt/instruction_action.txt")
    parser.add_argument("--one_shot_path", type=str, default="prompt/one_shot_action.txt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="# Summary Workflows")

    args = parser.parse_args()

    main()
