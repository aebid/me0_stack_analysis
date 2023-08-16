import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm

plot_prefix = "plots/awk_cluster/"

fname = "../input_files/rechit/awk_cluster_tree.root"

#Data looks like
#nclusterChamber = 7
#clusterChamber  = 0,
#                  0, 1, 1, 2, 2,
#                  3
#nclusterEta     = 7
#clusterEta      = 3,
#                  3, 3, 3, 3, 3,
#                  3
#nclusterStrip   = 7
#clusterStrip    = 5.5,
#                  35, 12, 27, 14, 19.5,
#                  11.5
#nclusterSize    = 7
#clusterSize     = 2,
#                  1, 5, 3, 3, 2, 4
#nclusterBX_first = 7
#clusterBX_first = 3,
#                  3, 2, 3, 2, 2,
#                  2
#nclusterBX_second = 7
#clusterBX_second = 3,
#                  3, 2, 3, 3, 3,
#                  2
#nclusterBX_last = 7
#clusterBX_last  = 3,
#                  3, 3, 4, 3, 3,
#                  3
#nclusterBX_avg  = 7
#clusterBX_avg   = 3,
#                  3, 2.4, 3.33333, 2.66667, 2.5,
#                  2.25


uproot_file = uproot.open(fname)
tree = uproot_file['clusterTree']
events = tree.arrays()


ch_list = [0,1,2,3]
eta_list = [1,2,3,4]
n_strip_bins = 3 #3 VFATs

BX_options = [[events.clusterBX_first, "First_BX"], [events.clusterBX_second, "Second_BX"], [events.clusterBX_last, "Last_BX"], [events.clusterBX_avg, "Average_BX"], [events.clusterBX_avg, "Average_BX_round"]]

one_hit_per_chamber = (ak.sum(events.clusterChamber == 0, axis=1) == 1) & (ak.sum(events.clusterChamber == 1, axis=1) == 1) & (ak.sum(events.clusterChamber == 2, axis=1) == 1) & (ak.sum(events.clusterChamber == 3, axis=1) == 1)
print("one hit test ", one_hit_per_chamber)
for bx, name in BX_options:
    print(name)
    strips = events.clusterStrip

    fig_occupancy, axs_occupancy = plt.subplots(4, 4, figsize=(20,20))

    fig_occupancy_one_per_chamber, axs_occupancy_one_per_chamber = plt.subplots(4, 4, figsize=(20,20))

    fig_timeres, axs_timeres = plt.subplots(4, figsize=(20,20))

    fig_timeres_hist, axs_timeres_hist = plt.subplots(4, figsize=(20,20))

    fig_BX_eta3_strips100_150, axs_BX_eta3_strips100_150 = plt.subplots(4, figsize=(20,20))

    for ch_num in ch_list:
        eta3_strips100_150_mask = (events.clusterChamber == ch_num) & (events.clusterEta == 3) & (strips >= 100) & (strips <= 150)
        if name == "Average_BX_round":
            x = np.round(ak.to_numpy(ak.flatten(bx[eta3_strips100_150_mask])), 0)
        else:
            x = ak.to_numpy(ak.flatten(bx[eta3_strips100_150_mask]))

        tmp_eta3_strips100_150_hist = axs_BX_eta3_strips100_150[ch_num].hist(x, bins=7, range=(0.0,7.0))
        axs_BX_eta3_strips100_150[ch_num].set_xlabel(name)
        axs_BX_eta3_strips100_150[ch_num].set_ylabel("Entries")
        axs_BX_eta3_strips100_150[ch_num].set_title("BX Ch{} Eta3 Strips[100,150]".format(ch_num))
        fig_BX_eta3_strips100_150.text(0.8, 0.25*((3-ch_num)+1)-0.05, "Sigma: {}".format(round(norm.fit(x)[1],3)), fontsize=24)


        x_time_res = np.empty([n_strip_bins*len(eta_list)])
        y_time_res = np.empty([n_strip_bins*len(eta_list)])
        y_time_res_err = np.empty([n_strip_bins*len(eta_list)])
        for eta_num in eta_list:
            ch_eta_mask = (events.clusterChamber == ch_num) & (events.clusterEta == eta_num)
            #ch_eta_mask = (events.clusterChamber == ch_num) & (events.clusterEta == eta_num) & one_hit_per_chamber
            if name == "Average_BX_round":
                x = np.round(ak.to_numpy(ak.flatten(bx[ch_eta_mask])), 0)
            else:
                x = ak.to_numpy(ak.flatten(bx[ch_eta_mask]))

            y = ak.to_numpy(ak.flatten(strips[ch_eta_mask]))

            tmp_occupancy = axs_occupancy[eta_num-1, ch_num].hist2d(x, y, bins=(10,400), range=((0,10),(0,400)), norm='log')
            fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=axs_occupancy[eta_num-1, ch_num])
            axs_occupancy[eta_num-1, ch_num].set_xlabel(name)
            axs_occupancy[eta_num-1, ch_num].set_ylabel("Digi Strip")
            axs_occupancy[eta_num-1, ch_num].set_title("Cluster DigiStrip vs {} Ch{} Eta{}".format(name, ch_num, eta_num))


            """
            one_cluster_per_chamber_mask = (ak.sum(events.clusterChamber == ch_num, axis=1) == 1)
            x = ak.to_numpy(ak.flatten(bx[ch_eta_mask & one_cluster_per_chamber_mask]))
            y = ak.to_numpy(ak.flatten(strips[ch_eta_mask & one_cluster_per_chamber_mask]))

            tmp_occupancy = axs_occupancy_one_per_chamber[eta_num-1, ch_num].hist2d(x, y, bins=(10,400), range=((0,10),(0,400)), norm='log')
            fig_occupancy_one_per_chamber.colorbar(tmp_occupancy[3], label="Number of Entries", ax=axs_occupancy_one_per_chamber[eta_num-1, ch_num])
            axs_occupancy_one_per_chamber[eta_num-1, ch_num].set_xlabel(name)
            axs_occupancy_one_per_chamber[eta_num-1, ch_num].set_ylabel("Digi Strip")
            axs_occupancy_one_per_chamber[eta_num-1, ch_num].set_title("Cluster DigiStrip vs {} Masked Ch{} Eta{}".format(name, ch_num, eta_num))
            """

            #The ideal BX for each eta over strip bins
            for strip_bin_num in range(n_strip_bins):
                strip_start = (strip_bin_num)*(128*3)/n_strip_bins
                strip_end = (strip_bin_num+1)*(128*3)/n_strip_bins
                #Look at all digiEvents on eta while in awkward
                if name == "Average_BX_round":
                    x = np.round(ak.to_numpy(ak.flatten(bx[ch_eta_mask])), 0)
                else:
                    x = ak.to_numpy(ak.flatten(bx[ch_eta_mask]))

                y = ak.to_numpy(ak.flatten(strips[ch_eta_mask]))

                #Only look at BX less than 7 and strip within bins in the flat numpy
                mask = (x < 7) & (y >= strip_start) & (y <= strip_end)
                x1 = x[mask]
                y1 = y[mask]

                (mu, sigma) = norm.fit(x1)
                mu = np.nan_to_num(mu, 0)
                sigma = np.nan_to_num(sigma, 0)
                x_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
                y_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
                y_time_res_err[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma

        tmp_time_res = axs_timeres[ch_num].errorbar(x_time_res, y_time_res, yerr=y_time_res_err)
        axs_timeres[ch_num].set_xlabel("Strip Bin (x4)")
        axs_timeres[ch_num].set_ylabel("Mean "+name)
        axs_timeres[ch_num].set_title("Time Resolution Ch{}".format(ch_num, eta_num))
        axs_timeres[ch_num].set_ylim(1,5)


        tmp_time_res_hist = axs_timeres_hist[ch_num].hist(y_time_res_err, bins=20, range=(0.5,1.0))
        axs_timeres_hist[ch_num].set_xlabel(name+" Sigma")
        axs_timeres_hist[ch_num].set_ylabel("Entries")
        axs_timeres_hist[ch_num].set_title("Time Resolution Ch{}".format(ch_num, eta_num))
        (mu, sigma) = norm.fit(y_time_res_err)
        mu = np.nan_to_num(mu, 0)
        sigma = np.nan_to_num(sigma, 0)
        fig_timeres_hist.text(0.8, 0.25*((3-ch_num)+1)-0.05, "Mean: {}".format(round(mu,3)), fontsize=24)
        #axs_timeres_hist[ch_num].set_xlim(0,1.5)


    fig_occupancy.tight_layout()
    fig_occupancy.savefig(plot_prefix+"occupancy_"+name+".pdf".format(ch_num, eta_num))
    plt.close(fig_occupancy)

    #fig_occupancy_one_per_chamber.tight_layout()
    #fig_occupancy_one_per_chamber.savefig(plot_prefix+"occupancy_masked_"+name+".pdf".format(ch_num, eta_num))

    fig_timeres.tight_layout()
    fig_timeres.savefig(plot_prefix+"time_res_"+name+".pdf")
    plt.close(fig_timeres)

    fig_timeres_hist.tight_layout()
    fig_timeres_hist.savefig(plot_prefix+"time_res_hist_"+name+".pdf")
    plt.close(fig_timeres_hist)

    fig_BX_eta3_strips100_150.tight_layout()
    fig_BX_eta3_strips100_150.savefig(plot_prefix+"BX_eta3_strip100_150_"+name+".pdf")
    plt.close(fig_BX_eta3_strips100_150)
