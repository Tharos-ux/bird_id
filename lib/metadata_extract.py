import pandas as pd
from argparse import ArgumentParser


def metadata_extract(input: str, output: str="metadata.json"):
    metadata = pd.read_csv(input)
    with open(output, "w") as file:
        file.write("{\n")
        for i, id in enumerate(metadata.filename):
            file.write(f"\"{str(id)}\" : {float(metadata.rating[i])}")
            if i != len(metadata.filename)-1:
                file.write(",")
            file.write("\n")
        file.write("}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", help="path to input .csv file", type=str)
    parser.add_argument("output", help="path to output .json file", type=str, nargs='?', default="metadata.json")
    args = parser.parse_args()

    metadata_extract(args.input, args.output)


