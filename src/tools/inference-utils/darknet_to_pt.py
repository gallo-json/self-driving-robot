import sys
sys.path.insert(1, '/home/jose/Programming/aiml/tools/yolov3-archive')

from models import convert
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='*.cfg path')
parser.add_argument('--weights', type=str, help='*.weights path')
args = parser.parse_args()

convert(args.cfg, args.weights)