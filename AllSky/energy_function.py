
import numpy as np
import matplotlib.pyplot as plt
import os

MCprihist = np.load("/home2/hky/github/Gamma_Energy/MCdata/priEvent.npy")
MCprihist_new = MCprihist[:-2]
MCprihist_new[-1] = MCprihist[-1]+MCprihist[-2]
MCprihist_new[-2] = MCprihist[-3]+MCprihist[-4]

# precision = dict()
precision = np.load(
    "/home2/hky/github/Gamma_Energy/precision.npy", allow_pickle=True).item()
Energy_min = np.logspace(0.2, 2.6, 13)
Energy_max = np.logspace(0.4, 2.8, 13)
bins_1 = np.linspace(-2, 2, 41)
bins_2 = np.linspace(0.2, 2.8, 14)
bins_3 = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8])
bins_2_center = (bins_2[:-1] + bins_2[1:]) / 2
bins_3_center = (bins_3[:-1] + bins_3[1:]) / 2


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def LIMA(alpha, Non, Noff):
    sig = np.sqrt(2) * np.sqrt(
        Non * np.log((1 + alpha) / alpha * (Non / (Non + Noff)))
        + Noff * np.log((1 + alpha) * Noff / (Non + Noff))
    )
    if type(sig) is np.ndarray:
        sig[np.where((Non - Noff * alpha) < 0)] = -sig[
            np.where((Non - Noff * alpha) < 0)
        ]
    else:
        if (Non - Noff * alpha) < 0:
            sig = -sig
    return sig


def check_fit(energy_pred, energy_orgin, title, savepath):
    plt.tight_layout()
    fig, axes = plt.subplots(5, 3, sharey=True, figsize=(16, 9))
    precision_ = list()
    axes = axes.reshape(-1)
    for i in range(13):
        need = np.where((Energy_min[i] < energy_pred)
                        & (Energy_max[i] > energy_pred))
        need_draw = np.log(energy_pred[need]/energy_orgin[need])
        precision_.append(np.std(need_draw))
        axes[i].hist(need_draw, bins=bins_1, density=True)
        axes[i].set_title(
            f"{Energy_min[i]:.2f}<E_pred<{Energy_max[i]:.2f}", fontsize=10)
        axes[i].set_ylim(0, 4)
        axes[i].text(-2, 3,
                     f"std={np.std(need_draw):.2f}\nmean={np.mean(need_draw):.2f}\n")
    plt.xlabel("ln(pred/true)")
    mkdir(savepath)
    plt.savefig(os.path.join(savepath, "resolution.png"))
    plt.show()

    hist_orgin, _ = np.histogram(np.log10(energy_orgin), bins=bins_2)
    hist_recon, _ = np.histogram(np.log10(energy_pred), bins=bins_2)

    plt.errorbar(
        bins_2_center,
        hist_orgin,
        np.sqrt(hist_orgin),
        np.ones(9) * 0.1,
        fmt=".",
        label="true Energy"
    )
    plt.errorbar(
        bins_2_center,
        hist_recon,
        np.sqrt(hist_recon),
        np.ones(9) * 0.1,
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

    plt.scatter(np.log10(energy_pred), np.log10(energy_orgin), s=1)
    plt.plot([0, 3], [0, 3], c="r")
    plt.xlabel("energy_pred")
    plt.ylabel("energy_orgin")
    plt.savefig(os.path.join(savepath, "compare.png"))
    plt.show()
    precision[title] = precision_


def getWindowsSize(sumpf):
    WindowsSize = 6.9/np.sqrt(sumpf)
    WindowsSize[WindowsSize > 1] = 1
    WindowsSize[WindowsSize < 0.5] = 0.5
    return WindowsSize


def getAngleDistance(theta, phi, pritheta, priphi):
    return np.rad2deg(np.arccos(np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(pritheta))*np.cos(np.deg2rad(priphi-phi))+np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(pritheta))))


def get_eta(energy_pred, sectheta, sumpf, theta, phi, pritheta, priphi, test_size):
    energy_pred = energy_pred.reshape(-1)
    AngleDistance = getAngleDistance(theta, phi, pritheta, priphi)
    WindowsSize = getWindowsSize(sumpf)
    MCafterhist = np.zeros(7)
    bins = 10**bins_3
    for i in range(7):
        indices = np.where((energy_pred > bins[i]) & (
            energy_pred < bins[i+1]) & (AngleDistance < WindowsSize))
        MCafterhist[i] = np.sum(1/sectheta[indices])
    eta = MCafterhist/MCprihist_new
    return eta/test_size


def GetOnOff(Exptdata, ExptEnergy):
    On = np.zeros(7)
    Off = np.zeros(7)
    # bins_mean = np.zeros(9)

    Ra_min = 81.5
    Ra_max = 85.5
    Ra_gep = 0.1
    x_binscount = int(np.round((Ra_max - Ra_min) / Ra_gep)) + 1

    Dec_min = 20
    Dec_max = 24
    Dec_gep = 0.1
    y_binscount = int((Dec_max - Dec_min) / Dec_gep) + 1

    x_bins = np.linspace(Ra_min, Ra_max, x_binscount + 1)
    y_bins = np.linspace(Dec_min, Dec_max, y_binscount + 1)
    x = (x_bins[:-1] + x_bins[1:]) / 2
    y = (y_bins[:-1] + y_bins[1:]) / 2

    for energybin in range(7):
        ALL = np.zeros([len(x), len(y)])
        Background = np.zeros([len(x), len(y)])

        index_energy_cut = np.where(
            (ExptEnergy > 10**bins_3[energybin])
            & (ExptEnergy < 10**bins_3[energybin+1])
            & (Exptdata["Dec"] > Dec_min-2)
            & (Exptdata["Dec"] < Dec_max + 2)
            & (Exptdata["Ra"] > Ra_min-2)
            & (Exptdata["Ra"] < Ra_max+2)
        )
        for i in range(x_binscount):
            for j in range(y_binscount):
                distance = getAngleDistance(
                    90 - Exptdata["Dec"][index_energy_cut],
                    Exptdata["Ra"][index_energy_cut],
                    90 - y[j],
                    x[i],
                )
                ALL[i, j] = np.sum(distance < getWindowsSize(
                    Exptdata["sumpf"][index_energy_cut]))

        for k in range(20):
            index_energy_cut = np.where(
                (ExptEnergy > 10**bins_3[energybin])
                & (ExptEnergy < 10**bins_3[energybin+1])
                & (Exptdata["DecOFF"][:, k] > Dec_min-2)
                & (Exptdata["DecOFF"][:, k] < Dec_max + 2)
                & (Exptdata["RaOFF"][:, k] > Ra_min-2)
                & (Exptdata["RaOFF"][:, k] < Ra_max+2)
            )
            for i in range(x_binscount):
                for j in range(y_binscount):
                    distance = getAngleDistance(
                        90 - Exptdata["DecOFF"][index_energy_cut, k],
                        Exptdata["RaOFF"][index_energy_cut, k],
                        90 - y[j],
                        x[i],
                    )
                    Background[i, j] += np.sum(distance < getWindowsSize(
                        Exptdata["sumpf"][index_energy_cut]))
        Background = Background.T
        for i in range(y_binscount):
            Background[i] = np.mean(Background[i])
        sig = LIMA(0.05, ALL, Background.T)
        sig[np.isnan(sig)] = 0

        sigmax = 0
        for i in range(15, 26):
            for j in range(15, 26):
                if sig[i, j] > sigmax:
                    sigmax = sig[i, j]
                    i_max = i
                    j_max = j
        lightest_x = x[i_max]
        lightest_y = y[j_max]
        # fig, ax = plt.subplots(facecolor="white")
        # c = ax.pcolormesh(x, y, sig.T, cmap="plasma")
        # ax.scatter(
        #     lightest_x, lightest_y, c="r", marker="*", label="this work(lightest)", s=80
        # )
        # ax.set_title(
        #     f"{10**bins_2[energybin]:.2f}<E<{10**bins_2[energybin+1]:.2f} Sky Map(RA:{lightest_x:.2f} DEC:{lightest_y:.2f})")
        On[energybin] = ALL[i_max, j_max]
        Off[energybin] = Background[i_max, j_max]
        # ax.set_xlabel("Ra(deg)")
        # ax.set_ylabel("Dec(deg)")
        # ax.legend(loc="upper left")
        # fig.colorbar(c, ax=ax)
        # ax.invert_xaxis()
        # plt.show()
    return On, Off


def GetSpectrum(Exptdata, ExptEnergy, energy_pred, sectheta, sumpf, theta, phi, pritheta, priphi, test_size, savepath):
    On, Off = GetOnOff(Exptdata, ExptEnergy)
    print(On)
    print(Off)
    over = On-Off*0.05
    rate = get_eta(energy_pred, sectheta, sumpf, theta,
                   phi, pritheta, priphi, test_size)
    Spectrum = over / 176.158 / 24 / 60 / 60 / rate / \
        (30000 * 30000 * 3.1415926) / \
        (10**bins_3[1:]-10**bins_3[:-1])*(10**bins_3_center)**2
    Spectrum_err = np.sqrt(over) / 176.158 / 24 / 60 / 60 / rate / \
        (30000 * 30000 * 3.1415926) / \
        (10**bins_3[1:]-10**bins_3[:-1])*(10**bins_3_center)**2
    prl_E = [10.69, 17.91, 30, 49, 75, 140, 350]
    prl_Spectrum = [8.5e-12, 6e-12, 4e-12, 2.1e-12, 1.4e-12, 4.8e-13, 2.18e-13]
    prl_err = [1e-12, 1e-12, 0.4e-12, 0.3e-12, 0.2e-12, 1.4e-13, 1.35e-13]
    plt.errorbar(prl_E, prl_Spectrum, prl_err, label="prl", fmt="o")
    print(10**bins_3_center, Spectrum,
          Spectrum_err, )
    plt.errorbar(10**bins_3_center, Spectrum,
                 Spectrum_err, label="this work", fmt="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("E(TeV)")
    plt.ylabel("SED")
    plt.legend()
    plt.savefig(os.path.join(savepath, "EnergySpectrum.png"))
    plt.show()


def draw_precision():
    for key in precision.keys():
        plt.plot((Energy_min+Energy_max)/2,
                 precision[key], "o:", label=key)
    plt.legend()
    plt.xlabel("Energy(TeV)")
    plt.ylabel("energy resolution")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.1, 0.5)
    plt.savefig("./precision.png")
