
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu-id',               default='0', type=str, metavar='N')
parser.add_argument('-e', '--encoder-architecture', default="resnet50", type=str, help="encoder architecture param for pytorch_segmentation_models")
parser.add_argument('-p', '--weights',              default='imagenet', type=str)
parser.add_argument('-r', '--resume',               default='', type=str, help="path to checkpoint to use to resume trainig")
parser.add_argument('-f', '--freeze-encoder',       default=False, type=bool)
parser.add_argument('-s', '--label-sigma',          default=1, type=float, help="gauss kernel sigma applied to label mask")
parser.add_argument('-d', '--label-value',          default=100, type=float, help="initialization value for pixel with annotation")
parser.add_argument('-bs', '--batch-size',          default=16, type=int)
parser.add_argument('-dr', '--diagnostic-run',      default=False, type=bool, help="whether to run just few iteration to see if everything is working")
args = parser.parse_args()


with open("provaset.json" , "w") as f:
    json.dump(vars(args), f, indent=4)