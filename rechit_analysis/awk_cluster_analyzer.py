import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm

plot_prefix = "plots/run386/"

fname = "../awk_cluster_run386.root"

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
n_strip_bins = int(128*3 / 4) #n bins per eta partition
n_stips_per_bin = 128*3/n_strip_bins


if not os.path.exists(plot_prefix):
    os.makedirs(plot_prefix)

BX_options = [[events.clusterBX_first, "First_BX"], [events.clusterBX_second, "Second_BX"], [events.clusterBX_last, "Last_BX"], [events.clusterBX_avg, "Average_BX"], [events.clusterBX_avg, "Average_BX_round"]]

one_hit_per_chamber = (ak.sum(events.clusterChamber == 0, axis=1) == 1) & (ak.sum(events.clusterChamber == 1, axis=1) == 1) & (ak.sum(events.clusterChamber == 2, axis=1) == 1) & (ak.sum(events.clusterChamber == 3, axis=1) == 1)
print("one hit test ", one_hit_per_chamber)
for bx, name in BX_options:
    print(name)
    print(bx)
    strips = events.clusterStrip

    fig_occupancy, axs_occupancy = plt.subplots(4, 4, figsize=(20,20))

    fig_occupancy_one_per_chamber, axs_occupancy_one_per_chamber = plt.subplots(4, 4, figsize=(20,20))

    fig_timeres, axs_timeres = plt.subplots(4, figsize=(20,20))

    fig_timeres_hist, axs_timeres_hist = plt.subplots(4, figsize=(20,20))

    #Figure for the time resolution split by eta partitions
    fig_timeres_hist_by_eta, axs_timeres_hist_by_eta = plt.subplots(4, 4, figsize=(20,20))


    for ch_num in ch_list:
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
                mask = (x < 7) & (x > 1) & (y >= strip_start) & (y <= strip_end)
                x1 = x[mask]
                y1 = y[mask]

                (mu, sigma) = norm.fit(x1)
                mu = np.nan_to_num(mu, 0)
                sigma = np.nan_to_num(sigma, 0)
                x_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
                y_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
                y_time_res_err[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma


            y_time_res_err_eta = y_time_res_err[(eta_num-1)*n_strip_bins:(eta_num)*n_strip_bins]
            tmp_time_res_hist_by_eta = axs_timeres_hist_by_eta[eta_num-1, ch_num].hist(y_time_res_err_eta, bins=50, range=(0.2,1.0))
            axs_timeres_hist_by_eta[eta_num-1, ch_num].set_xlabel("BX Sigma")
            axs_timeres_hist_by_eta[eta_num-1, ch_num].set_ylabel("Entries")
            axs_timeres_hist_by_eta[eta_num-1, ch_num].set_title("Time Resolution Ch{} Eta{}".format(ch_num, eta_num))
            y_time_res_err_eta_nonan = y_time_res_err_eta[~np.isnan(y_time_res_err_eta)]
            y_time_res_err_eta_nonan = y_time_res_err_eta_nonan[~(y_time_res_err_eta_nonan < 0.1)]
            fig_mean_by_eta = round(norm.fit(y_time_res_err_eta_nonan)[0],3)
            axs_timeres_hist_by_eta[eta_num-1, ch_num].text(0.5, 0.8, "Mean: {}".format(fig_mean_by_eta), fontsize=18, transform=axs_timeres_hist_by_eta[eta_num-1, ch_num].transAxes)




        tmp_time_res = axs_timeres[ch_num].errorbar(x_time_res, y_time_res, yerr=y_time_res_err)
        axs_timeres[ch_num].set_xlabel("Strip Bin (x{})".format(n_stips_per_bin))
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

    fig_timeres.tight_layout()
    fig_timeres.savefig(plot_prefix+"time_res_"+name+".pdf")
    plt.close(fig_timeres)

    fig_timeres_hist.tight_layout()
    fig_timeres_hist.savefig(plot_prefix+"time_res_hist_"+name+".pdf")
    plt.close(fig_timeres_hist)

    fig_timeres_hist_by_eta.tight_layout()
    fig_timeres_hist_by_eta.savefig(plot_prefix+"time_res_hist_by_eta_"+name+".pdf")
    plt.close(fig_timeres_hist_by_eta)
