
import numpy as np
import matplotlib.pyplot as plt
import os


# precision = dict()
precision = np.load(
    "/home2/hky/github/Gamma_Energy/AllSky/precision_ALLSKY_sigma<1.npy", allow_pickle=True).item()
Energy_bin = np.logspace(0.6, 4.6, 21)
Energy_bin_center = 10**((np.log10(Energy_bin[:-1]) +
                         np.log10(Energy_bin[1:]))/2)
Energy_min = Energy_bin[:-1]
Energy_max = Energy_bin[1:]
bins_1 = np.linspace(-2, 2, 41)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_fit(energy_pred, energy_orgin, title, savepath):
    plt.tight_layout()
    _, axes = plt.subplots(4, 5, sharey=True, figsize=(15, 15))
    precision_ = list()
    axes = axes.reshape(-1)
    for i in range(len(Energy_bin)-1):
        need = np.where((Energy_min[i] < energy_pred)
                        & (Energy_max[i] > energy_pred))
        need_draw = np.log(energy_pred[need]/energy_orgin[need])
        precision_.append(np.sqrt(np.mean(need_draw**2)))
        axes[i].hist(need_draw, bins=bins_1, density=True)
        axes[i].set_title(
            f"{Energy_min[i]:.2f}<E_pred<{Energy_max[i]:.2f}", fontsize=10)
        axes[i].set_ylim(0, 4)
        axes[i].text(-2, 3,
                     f"error={np.sqrt(np.mean(need_draw**2)):.2f}\nstd={np.std(need_draw):.2f}\nmean={np.mean(need_draw):.2f}\n")
    plt.xlabel("ln(pred/true)")
    mkdir(savepath)
    plt.savefig(os.path.join(savepath, "resolution.png"))
    plt.show()
    plt.close()
    hist_orgin, _ = np.histogram(
        np.log10(energy_orgin), bins=np.log10(Energy_bin))
    hist_recon, _ = np.histogram(
        np.log10(energy_pred), bins=np.log10(Energy_bin))

    plt.errorbar(
        np.log10(Energy_bin_center),
        hist_orgin,
        np.sqrt(hist_orgin),
        np.ones(len(Energy_bin_center)) * 0.1,
        fmt=".",
        label="true Energy"
    )
    plt.errorbar(
        np.log10(Energy_bin_center),
        hist_recon,
        np.sqrt(hist_recon),
        np.ones(len(Energy_bin_center)) * 0.1,
        fmt=".",
        label="pred Energy"
    )
    plt.title(title)
    plt.xlabel("log10energy")
    plt.ylabel("dN")
    plt.legend()
    plt.yscale("log")
    plt.savefig(os.path.join(savepath, "compare_bin.png"))
    plt.show()
    plt.close()

    plt.scatter(np.log10(energy_pred), np.log10(energy_orgin), s=1)
    plt.plot([0, 5], [0, 5], c="r")
    plt.xlabel("energy_pred")
    plt.ylabel("energy_orgin")
    plt.savefig(os.path.join(savepath, "compare.png"))
    plt.show()
    precision[title] = precision_
    plt.close()


def draw_precision():
    for key in precision.keys():
        plt.plot(Energy_bin_center,
                 precision[key], "o:", label=key)
    plt.legend()
    plt.xlabel("Energy(TeV)")
    plt.ylabel("energy resolution")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.1, 0.6)
    plt.savefig("./precision.png")
    plt.show()
