
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


legend_elements = [
    Patch(facecolor="skyblue", label="White"),
    Patch(facecolor="tomato", label="Red")
]

def plot_box(white, red, title, save_loc):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=white, orient="h", color="skyblue")
    sns.boxplot(data=red, orient="h", color="tomato")
    plt.axvline(x=0, color="gray", ls="--")
    plt.xlabel("Coefficients")
    plt.title(title)
    plt.legend(handles=legend_elements, title="Wine Type", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_loc)