import argparse
from easytorch.utils.defaultargs import ap
import dataspecs as dspec

from easytorch import EasyTorch
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)
dataspecs = [dspec.DRIVE, dspec.STARE]
runner = EasyTorch(ap, dataspecs)

if __name__ == "__main__":
    runner.run(MyDataset, MyTrainer)
    runner.run_pooled(MyDataset, MyTrainer)
