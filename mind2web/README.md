# ATM for Mind2Web

## Install

```bash
pip install -r requirements.txt
```

Download data from the [mind2web](https://github.com/OSU-NLP-Group/Mind2Web) project, make sure you have `test_task`, `test_website`, `test_domain`, and `train` under the `data` directory.

## Offline Workflow Induction + Test Inference

To run offline workflow induction with training examples:
```bash
python offline_induction.py \
--mode auto --domain Travel --subdomain Airlines --website aa \
--model "gpt-4o" --output_dir "workflow"
```
You can also switch to `--mode input` to dynamically input your desired website(s).

The above command will produce a workflow file `workflow/aa.txt`, to augment this workflow in agent memory and run inference on test examples from the *aa* website:

```bash
python run_mind2web.py --website "aa" --workflow_path "workflow/aa.txt"
```

## Online Induction with Test Queries

To run online workflow induction and utilization:
```bash
python pipeline.py --setup online \
--benchmark "test_task" --website aa \
--results_dir results/aa/workflow \
--workflow_path workflow/aa.txt
```

Simply change to `--benchmark 'train'` if you want to run online setting on the training (or other) queries, but remember to apply to workflow and run inference on test examples afterwards.


## Overall
To run the entire pipeline for both online and offline settings, you can use:
```bash
python pipeline.py --setup "offline" # or "online"
```
with other arguments specified as above.
