**Introduction:**

This code is the supporting material for the article "BalCon -- resource balancing algorithm for VM consolidation".

**Installation:**

To run an example, you should install required packages with:

```
pip install -r requirements.txt
```

**Examples:**

To execute algorithms from command line use:

```
python run.py --problem ../dataset/000.json --algorithm balcon firstfit sercon-modified --tl 60
```

To perform your own experiments, take a look at `run_example` function in `run.py`, for instance:

```
run_example(problem_path='../dataset/000.json', algorithm=registry['balcon'], time_limit=60, wa=1, wm=2)
```

**Dataset:**

Synthetic dataset was generated with `generate.py`.
