"""Print recent native PyTorch training records."""
import argparse
from collections import deque
import json

parser = argparse.ArgumentParser()
parser.add_argument("metrics", help="Path to metrics.jsonl")
parser.add_argument("--last", type=int, default=10)
args = parser.parse_args()
with open(args.metrics) as records:
    for line in deque(records, maxlen=args.last):
        print(json.dumps(json.loads(line), indent=2))
