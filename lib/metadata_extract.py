import pandas as pd
from argparse import ArgumentParser
from json import dump


def extract_rating(input: str, output: str = "rating.json"):
    """Extract rating infos from metadata .csv into .json
    Args:
        input (str): metadata file
        output (str, optional): file path to store rating infos. Defaults to "rating.json".
    """
    metadata = pd.read_csv(input)
    dump({id: metadata.rating[i] for i, id in enumerate(
        metadata.filename)}, open(output, "w"))


def extract_name(input: str, output: str = "spec_name.json"):
    """Extract specie name infos from metadata .csv into .json
    Args:
        input (str): metadata file
        output (str, optional): file path to store name infos. Defaults to "spec_name.json".
    """
    metadata = pd.read_csv(input)
    code_name = {}
    for i, ebird_code in enumerate(metadata.ebird_code):
        try:
            code_name[ebird_code]
        except:
            code_name[ebird_code] = metadata.species[i]

    dump(code_name, open(output, "w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="path to input .csv file", type=str)
    parser.add_argument("output", help="path to output .json file", type=str)
    parser.add_argument("--rating", help="get rating", action="store_true")
    parser.add_argument("--name", help="get names", action="store_true")
    args = parser.parse_args()

    if args.rating:
        extract_rating(args.input, args.output)

    if args.name:
        extract_name(args.input, args.output)
