import os
import subprocess

def main(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    for i in range(3):
        cmd = [
            'python',
            'experiments.py',
            '--name={}'.format(args.logdir)]
        subprocess.run(cmd)

if __name__ == "__main__":
    main(argumentparser())
