import os
import json
import argparse
import matplotlib.pyplot as plt

def get_average(score_list: list[float], percentage: bool = False) -> float:
    score = sum(score_list) / len(score_list)
    return score * 100 if percentage else score


def main():
    files = os.listdir(args.results_dir)
    file_paths = [os.path.join(args.results_dir, f) for f in files]
    ele_acc, act_f1, step_sr, sr = [], [], [], []
    for fp in file_paths:
        res = json.load(open(fp, 'r'))[-1]
        ele_acc.append(get_average(res["element_acc"]))
        act_f1.append(get_average(res["action_f1"]))
        step_sr.append(get_average(res["step_success"]))
        sr.append(get_average(res["success"]))
    
    print(f"Element Acc: {get_average(ele_acc, True):5.1f}")
    print(f"Action F1  : {get_average(act_f1, True):5.1f}")
    print(f"Step SR    : {get_average(step_sr, True):5.1f}")
    print(f"SR         : {get_average(sr, True):5.1f}")

    # accumulative step success rate
    n = len(step_sr)
    x = [i+1 for i in range(n)]
    asr = [get_average(step_sr[:i+1]) for i in range(n)]
    plt.plot(x, asr)

    # moving average
    # window_size = 5
    # x, mavg = [], []
    # for i in range(n-window_size+1):
    #     x.append(i)
    #     mavg.append(get_average(step_sr[i:i+window_size]))
    # plt.plot(x, mavg)

    if args.viz_path is not None:
        plt.savefig(args.viz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--viz_path", type=str, default=None)
    args = parser.parse_args()

    main()
