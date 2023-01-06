from os import listdir, system
from pathlib import Path
from random import shuffle
from collections import Counter

spectros_path = "train_unfiltered"
sample_rate = 500

species = Counter({my_dir: listdir(f"{spectros_path}/{my_dir}")
                   for my_dir in listdir(spectros_path)})
print(Counter({my_dir: len(listdir(f"{spectros_path}/{my_dir}"))
               for my_dir in listdir(spectros_path)}))


for _, val in species.items():
    shuffle(val)

for sp in listdir(spectros_path):
    if len(species[sp]) >= sample_rate:
        Path(f"{spectros_path}_sampled/{sp}").mkdir(parents=True, exist_ok=True)
        for i in range(sample_rate):
            system(
                f"cp {spectros_path}/{sp}/{species[sp][i]} {spectros_path}_sampled/{sp}/")
