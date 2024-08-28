import os, json, random
import numpy as np
from pathlib import Path
from openai import BadRequestError
from utils.env import *
from utils.llm import (
    generate_response, num_tokens_from_messages,
    MAX_TOKENS, extract_from_response,
)

import logging
logger = logging.getLogger(__name__)


def get_exemplars(args) -> list:
    """Get exemplar workflows in the prompt."""
    # workflow memory
    memory = []
    workflow_text = open(args.workflow_path, 'r').read().strip()
    if len(workflow_text):
        memory = [[{"role": "user", "content": workflow_text}]]

    # concrete examples
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        concrete_examples = json.load(f)
    if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples 
            if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain, args.website]])
        ]
    elif any([args.subdomain in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples 
            if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain]])
        ]

    memory += random.sample(concrete_examples, 
        min(args.retrieve_top_k, len(concrete_examples)))
    memory = [[{k:v for k,v in m.items() if k!="specifier"} for m in e] for e in memory]
    return memory


def eval_sample(task_id, args, sample):
    # initialize metrics
    element_acc, action_f1, step_success, success = [], [], [], []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    exemplars = get_exemplars(args)
    # print(exemplars)

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]

    prev_actions, prev_obs = [], []
    previous_k = 5

    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]

        # get query, obs, act
        target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)
        # Continue next loop if the ground truth element is not in the cleaned html
        if len(pos_candidates) == 0:
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
            conversation.append("The ground truth element is not in cleaned html")
            continue

        # construct query
        query = []
        for o, a in zip(prev_obs, prev_actions):
            if len(query) == 0:
                query.append({
                    "role": "user",
                    "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n" + o,
                })
            else:
                query.append({"role": "user", "content": o})
            query.append({"role": "assistant", "content": a})
        
        obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
        if len(query) == 0:
            query.append({
                "role": "user",
                "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                + "Observation: `" + obs + "`",
            })
        else:
            query.append({"role": "user", "content": "Observation: `" + obs + "`"})
        
        prev_obs.append("Observation: `" + target_obs + "`")
        prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
        
        # token limit
        total_num_tokens = num_tokens_from_messages(sys_message + query, args.model)
        if total_num_tokens > MAX_TOKENS[args.model]:
            logger.info(
                f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[args.model]}), skipping..."
            )
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            conversation.append(
                {
                    "input": sys_message + query,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            continue

        # message
        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = num_tokens_from_messages(
                sys_message + demo_message + e + query, args.model
            )
            if total_num_tokens > MAX_TOKENS[args.model]:
                logger.info(
                    f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                )
                break
            else:
                demo_message.extend(e)

        message = sys_message + demo_message + query
        try:
            response, info = generate_response(
                messages=message,
                model=args.model,
                temperature=args.temperature,
                stop_tokens=["Task:", "obs:"],
            )
        except BadRequestError:
            response = ""
            info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    log_dir = Path(f"{args.log_dir}/{args.model}/{args.benchmark}/{args.website}/{args.suffix}")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)
