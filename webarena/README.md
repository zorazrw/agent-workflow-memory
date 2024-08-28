# ATM for WebArena

## Install

*Install `browsergym`*: Follow the instructions in [this README](https://github.com/ServiceNow/BrowserGym) to install `browsergym`.

```bash
pip install browsergym
playwright install chromium
```

*Setup the `webarena` specifics*:

```bash
pip install browsergym-webarena
python -c "import nltk; nltk.download('punkt')"
```

Set up the web servers and environment URLs (find more details in the [webarena readme](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)).

```bash
BASE_URL=<YOUR_SERVER_URL_HERE>
export WA_SHOPPING="$BASE_URL:7770/"
export WA_SHOPPING_ADMIN="$BASE_URL:7780/admin"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
export WA_WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"
```

Then, generate the config files for each task, which will be used during workflow induction.

```bash
cd config_files
python generate_test_data.py
cd ../
```

*Install agent and evaluation requirements*:

```bash
pip install -r requirements.txt  # agent
pip install -r autoeval/requirements.txt  # model-based evaluation
```

*Setup `openai` keys*

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```


## Run Agent

### Baseline Agent: No Memory

```bash
python run.py --task "webarena.0" # switch task id from 0 to 811
```

You can check the results in `./results/...webarena.0.../`.

### with Agent Task Memory :atm:

**Step 1**. To run inference on a task:

```bash
python run.py --task "webarena.0" \
--workflow_path "workflow/shopping.txt"
```

Remember to match the workflow path with the associated tasks of the input id. By default, name the workflow file with the website name, e.g., 'shopping_admin.txt', 'reddit.txt', 'gitlab.txt', 'map.txt'.

**Step 2**. To evaluate an agent-generated task trajectory, run:

```bash
python -m autoeval.evaluate_trajectory --result_dir "../results/webarena.0"
```

This will produce a "{model}_autoeval.txt" in the "../results/webarena.0/" directory.

Change the `model` and `prompt` format options if necessary.

**Step 3**. Integrate the trajectory workflows to agent memory

```bash
python workflow_induction.py --results_dir results/shopping
```

Switch the `criteria` to "gt" (and can skip step 2) if you want to use ground-truth reward as signals to integrate workflows.

Iterate the loop of steps 1-3 for each task, or every n tasks to your demand.
We provide a pipeline script to iteratively execute the above steps for all tasks:

```bash
python pipeline.py --website "shopping"
```
