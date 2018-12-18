import subprocess
import numpy as np

def main(iters):
    lrs = np.exp(-np.linspace(1, 12, iters))
    for i, lr in zip(range(iters), lrs):
        fn_call = [
            "python",
            "lr2rl.py",
            "--trial={}".format(i),
            "--name={}".format(lr),
            "--lr={}".format(lr),
            "--logdir=/local/scratch/lr2rl",
            ]
        print(fn_call)
        subprocess.call(fn_call)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_experiments", type=int, default=20, help="the number of experiments")
    args = parser.parse_args()
    main(args.n_experiments)
