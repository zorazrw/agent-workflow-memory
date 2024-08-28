"""Online Induction and Workflow Utilization Pipeline."""

import argparse
import subprocess
from utils.data import load_json

def offline():
    # workflow induction
    process = subprocess.Popen([
        'python', 'offline_induction.py',
        '--mode', 'auto', '--website', args.website,
        '--domain', args.domain, '--subdomain', args.subdomain,
        '--model', args.model, '--output_dir', "workflow",
        '--instruction_path', args.instruction_path,
        '--one_shot_path', args.one_shot_path,
    ])
    process.wait()

    # test inference
    process = subprocess.Popen([
        'python', 'run_mind2web.py',
        '--website', args.website,
        '--workflow_path', f"workflow/{args.website}.txt"
    ])
    process.wait()


def online():
    # load all examples for streaming
    samples = load_json(args.data_dir, args.benchmark)
    print(f"Loaded #{len(samples)} test examples")
    if args.website is not None:
        samples = [s for s in samples if s["website"] == args.website]
        print(f"Filtering down to #{len(samples)} examples on website [{args.website}]")
    n = len(samples)
    
    for i in range(0, n, args.induce_steps):
        j = min(n, i + args.induce_steps)
        print(f"Running inference on {i}-{j} th example..")

        process = subprocess.Popen([
            'python', 'run_mind2web.py',
            '--benchmark', args.benchmark,
            '--workflow_path', args.workflow_path,
            '--website', args.website, 
            '--start_idx', f'{i}', '--end_idx', f'{j}',
            '--domain', args.domain, '--subdomain', args.subdomain,
        ])
        process.wait()
        print(f"Finished inference on {i}-{j} th example!\n")

        if (j + 1) < len(samples):
            process = subprocess.Popen([
                'python', 'online_induction.py',
                '--benchmark', args.benchmark,
                '--website', args.website,
                '--results_dir', args.results_dir,
                '--output_path', args.workflow_path,
            ])
            process.wait()
            print(f"Finished workflow induction with 0-{i} th examples!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # examples
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--website", type=str, required=True)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--subdomain", type=str, default=None)

    # results and workflows
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--workflow_path", type=str, default=None)

    # prompt
    parser.add_argument("--instruction_path", type=str, default="prompt/instruction_action.txt")
    parser.add_argument("--one_shot_path", type=str, default="prompt/one_shot_action.txt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="# Summary Workflows")

    # gpt
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=str, default=0.0)

    # induction frequency
    parser.add_argument("--induce_steps", type=int, default=1)

    # setup
    parser.add_argument("--setup", type=str, required=True,
                        choices=["online", "offline"])

    args = parser.parse_args()

    if args.setup == "online":
        assert (args.results_dir is not None) and (args.workflow_path is not None)
        online()
    elif args.setup == "offline":
        assert (args.domain is not None) and (args.subdomain is not None)
        offline()
