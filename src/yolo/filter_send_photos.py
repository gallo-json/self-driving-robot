import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("from_dir", help="Directory with the pictures and the txt files", type=str)
parser.add_argument("train_dir", help="train dir", type=str)
parser.add_argument("test_dir", help="test_dir", type=str)
parser.add_argument("train_split", help="train test split (in decimal), value for train", type=float)
args = parser.parse_args()

for file in os.listdir(args.from_dir):
    name = file[:-4]

    if os.path.exists(os.path.join(args.from_dir, name + '.txt')) and os.path.exists(os.path.join(args.from_dir, name + '.jpg')):
        if random.random() < args.train_split:
            os.system('mv ' + os.path.join(args.from_dir, name + '.txt') + ' ' + args.train_dir)
            os.system('mv ' + os.path.join(args.from_dir, name + '.jpg') + ' '+ args.train_dir)
        else:
            os.system('mv ' + os.path.join(args.from_dir, name + '.txt') + ' '+ args.test_dir)
            os.system('mv ' + os.path.join(args.from_dir, name + '.jpg') + ' '+ args.test_dir)

