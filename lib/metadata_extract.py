import pandas as pd
from argparse import ArgumentParser
from json import dump


def metadata_extract(input: str, output: str="metadata.json"):
    metadata = pd.read_csv(input)
    dump({id:metadata.rating[i] for i, id in enumerate(metadata.filename)},open(output, "w"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="path to input .csv file", type=str)
    parser.add_argument("output", help="path to output .json file", type=str, nargs='?', default="metadata.json")
    args = parser.parse_args()

    metadata_extract(args.input, args.output)


