from os import listdir
from json import load
from numpy import asarray, std, mean
import matplotlib.pyplot as plt

# on met tout le début du nom des dossiers ici
# il faut qu'ils soient différenciables
# un nom par truc dans liste préfixe
NOMS = ["4, brut", "6, brut", "4, filtré", "6, filtré"]
PREFIXES: list = ["4L_non_filtre", "6L_non_filtre", "4L_filtre", "6L_filtre"]

# on met le nom de la métrique que l'on veut récup (key du dictionnaire json)
METRIC: str = 'true_epochs'
EPOCH_BOUND: int = 30

metric: list = [[] for _ in range(len(PREFIXES))]

# on donne le dir des modèles
PATH: str = "modeles_30sp"


def match(dir: str, prefixes: list) -> bool:
    for elt in prefixes:
        if elt in dir:
            return True
    return False


def get_prefix(dir: str, prefixes: str) -> str:
    for elt in prefixes:
        if elt in dir:
            return elt


is_metric_list: bool = False
x = [i for i in range(EPOCH_BOUND)]
plt.rcParams["figure.figsize"] = (6, 8)
for dir in [d for d in listdir(f"{PATH}/") if match(d, PREFIXES)]:
    dic_params: dict = load(open(f"{PATH}/{dir}/params.json", "r"))
    if isinstance(dic_params[METRIC], list):
        is_metric_list = True
        metric[PREFIXES.index(get_prefix(dir, PREFIXES))
               ].append(dic_params[METRIC][:EPOCH_BOUND])
    else:
        metric[PREFIXES.index(get_prefix(dir, PREFIXES))
               ].append(dic_params[METRIC])

if is_metric_list:
    for i, met in enumerate(metric):
        plt.errorbar(x, [mean(serie) for serie in asarray(
            met).transpose()], yerr=[std(serie)/2 for serie in asarray(
                met).transpose()], label=NOMS[i], fmt='--o')
    plt.legend()  # bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0
    plt.savefig(f"{METRIC}.png", transparent=True)
    plt.show()
else:
    error = [std(serie)/2 for serie in asarray(metric)]
    for i, val in enumerate([mean(serie) for serie in asarray(metric)]):
        plt.bar(i, val, yerr=error[i], label=NOMS[i],
                align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks([i for i in range(len(PREFIXES))], NOMS)
    plt.savefig(f"{METRIC}.png", transparent=True, bbox_inches='tight')
    plt.show()
