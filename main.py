import argparse

import dataspecs as dspec
from easytorch.utils.defaultargs import ap
from easytorch.runs import run, pooled_run
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.AV_WIDE, dspec.VEVIO]
if __name__ == "__main__":
    run(ap, dataspecs, MyDataset, MyTrainer)
    pooled_run(ap, dataspecs, MyDataset, MyTrainer)
