from models import MergeBart
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--enc-type', type=str, choices=['syn-pool', 'pool'])
parser.add_argument('--n-epochs', type=int, default=3)
ARGS = parser.parse_args()

model = MergeBart(ARGS.enc_type)

def train():
    pass

for epoch_num in range(ARGS.n_epochs):
    train()
