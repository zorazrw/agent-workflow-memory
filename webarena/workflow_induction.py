import os
import json
import random
import argparse


def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    blocks, block = [], []
    for line in open(path, 'r'):
        if line.strip() == "":
            blocks.append(block)
            block = []
        else:
            if line.strip():
                block.append(line.strip())
    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str and type(eval(arg[1:-1])) == int:
                    valid_actions.append(a)
            except:
                continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions

def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file."""
    blocks = load_blocks(path)
    think_list, action_list = [], []
    for i in range(1, len(blocks), 2):
        # action
        b = blocks[i]
        actions = remove_invalid_steps(b[1:])
        if len(actions) == 0: continue
        action_list.append(actions)
        # think
        b = blocks[i-1]
        idx = b[-1].index("browsergym.experiments.loop - INFO -")
        think_list.append(b[-1][idx+36: ].strip())
    
    assert len(think_list) == len(action_list)
    
    # TODO: merge same actions
    return think_list, action_list

def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        acts = '\n'.join(a)
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def get_abstract_trajectory(action_list: list[list[str]]) -> str:
    abstract = []
    for acts in action_list:
        for a in acts:
            s = a.index("(")
            e = a.index(',', s) if ',' in a[s:] else a.index(")", s)
            action = a[:s]
            if action != "send_msg_to_user":
                arg = a[s+1: e]
                abstract.append(f"{action}({arg})")
            else:
                abstract.append(f"{action}")
    return '_'.join(abstract)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, n)]


def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    args.result_dir = args.result_dir.split()
    if args.criteria == "gt":
        file_dirs = [
            os.path.join(res_dir, f) for res_dir in args.result_dir for f in os.listdir(res_dir) 
            if json.load(
                open(os.path.join(res_dir, f, "summary_info.json"))
            )["cum_reward"]
        ]
    elif args.criteria == "autoeval":
        file_dirs = []
        for res_dir in args.result_dir:
            for f in os.listdir(res_dir):
                record_path = os.path.join(res_dir, f, f"{args.model}_autoeval.json")
                if not os.path.exists(record_path): continue
                record = json.load(open(record_path))
                if record[0]["rm"]:
                    file_dirs.append(os.path.join(res_dir, f))
    else:
        raise ValueError(f"Invalid criteria: {args.criteria}.")
    
    print(f"Collected {len(file_dirs)} result directories.")

    # template id based deduplication
    template_dict = {}
    for f in file_dirs:
        # get query -> task objective
        task_id = f.split('/')[-1].split("_")[0].split(".")[1]
        config_path = os.path.join("config_files", f"{task_id}.json")
        config = json.load(open(config_path))
        query = config["intent"]

        template_id = config["intent_template_id"] # for deduplication

        # parse trajectory
        log_path = os.path.join(f, "experiment.log")
        try:
            think_list, action_list = extract_think_and_action(log_path)
        except:
            continue

        # add to template dict
        wdict = {"query": query, "think_list": think_list, "action_list": action_list}
        if template_id not in template_dict: template_dict[template_id] = []
        template_dict[template_id].append(wdict)
    selected_workflows = random_group_sample(template_dict, 1)
    print(f"#{len(selected_workflows)} result dirs after template dedup..")
    
    # deduplicate by abstract trajectory
    abstraj_dict = {}
    for w in selected_workflows:
        abs_traj = get_abstract_trajectory(w['action_list'])
        if abs_traj not in abstraj_dict:
            abstraj_dict[abs_traj] = []
        abstraj_dict[abs_traj].append(w)
    selected_workflows = random_group_sample(abstraj_dict, 1)
    print(f"#{len(selected_workflows)} result dirs after trajectory dedup..")

    # manual inspection
    def get_workflow(d: dict) -> str:
        return f"Query: {d['query']}\n" + format_trajectory(d['think_list'], d['action_list'])
    manual_workflows = []
    for w in selected_workflows:
        w = get_workflow(w)
        if args.auto: 
            to_add = 'y'
        else:
            to_add = input("Workflow: \n" + w + "\n\nAdd? (y/n): ")
        if to_add == 'y':
            manual_workflows.append(w)
    print(f"#{len(manual_workflows)} result dirs after manual inspection..")



    if args.output_path is None:
        website = config["sites"][0]  # assumes all results are about the same website
        args.output_path = f"workflow/{website}.txt"
        print(f"[Warning] no output path specified, using '{args.output_path}' by default")
        
    with open(args.output_path, 'w') as fw:
        fw.write('\n\n\n'.join(["## Concrete Examples"] + manual_workflows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to the output file.")
    parser.add_argument("--criteria", type=str, default="autoeval", 
                        choices=["gt", "autoeval"],
                        help="'gt': only use examples with gold reward, 'autoeval': use examples with autoeval reward.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o"])
    parser.add_argument("--auto", action="store_true", help="w/o manual workflow inspections.")
    args = parser.parse_args()

    main()
