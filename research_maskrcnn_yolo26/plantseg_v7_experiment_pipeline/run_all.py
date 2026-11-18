import argparse, subprocess, sys

def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--batch-size",type=int,default=4)
    args=ap.parse_args()

    py=[sys.executable, "-u"]
    run(py + ["01_prepare_v7.py"])
    run(py + ["02_smoke_test.py"])

    for t in [1,2,3,4]:
        run(py + ["03_train.py","--tier",str(t),"--epochs",str(args.epochs),
             "--batch-size",str(args.batch_size)])

    for t in [1,2,3,4]:
        run(py + ["04_evaluate.py","--tier",str(t)])

    run(py + ["05_benchmark.py","--all"])
    run(py + ["06_visualize.py","--num-images","6"])

if __name__=="__main__":
    main()
