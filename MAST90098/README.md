## Group 5 Project Genetic Algorithm for Makespan scheduling with knowledge based operator
Author: Wenrui Yuan, Chayasin Sae-Tia, Yaning Ma

Please find `requirements.txt` to install required package, which wil also be listed here:
    - numpy
    - pandas
    - tqdm


### Usage:

Run instances in `runner.py`, the process begins with first generating multiple instances, where you can also modify instance sizes in the `generate_instances(nums=10, save=False)` method.

### Format:
Both `GeneticMakespan` and `GreedyMakespan` takes a tuple of elements as an input (`instance=(n, m, jobs)`) and outputs a nother tuple of elements `T, C_max, time()-t0`.
