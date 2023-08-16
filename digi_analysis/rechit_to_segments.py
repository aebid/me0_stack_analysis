import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm
import ROOT
from array import array

plot_prefix = "plots/segments/"
#plot_prefix = "plots/run146/"

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


out_file = uproot.recreate("segment_tree.root")

#Goal is to make segments from existing clusters
#Option to use only 3 layers and propagate to 4th (efficiency) or full 4 cluster segments (time res)
#Starting with find BEST segment in the entire event

ch_list = [0,1,2,3]
eta_list = [1,2,3,4]


nEvents = len(events)
print("There are ", nEvents, " events")

ch = events.clusterChamber
eta = events.clusterEta
strip = events.clusterStrip
strip_error = events.clusterSize * (1/(12**(0.5)))
BX_first = events.clusterBX_first

fig_occupancy, axs_occupancy = plt.subplots(4, 4, figsize=(20,20))
residual_axis_range = ((0,128*3), (-1,1))
residual_bins = (16, 10)

bx_fig_occupancy, bx_axs_occupancy = plt.subplots(4, 4, figsize=(20,20))
bx_axis_range = ((0,128*3), (0,5))
bx_bins = (16, 5)


clustAvgFl_bx_fig_occupancy, clustAvgFl_bx_axs_occupancy = plt.subplots(4, figsize=(20,20))
clustAvgInt_bx_fig_occupancy, clustAvgInt_bx_axs_occupancy = plt.subplots(4, figsize=(20,20))
clustFirst_bx_fig_occupancy, clustFirst_bx_axs_occupancy = plt.subplots(4, figsize=(20,20))
clustLast_bx_fig_occupancy, clustLast_bx_axs_occupancy = plt.subplots(4, figsize=(20,20))
clust_bx_axis_range = ((0,128*3), (0,5))
clust_bx_bins = (32, 15)



n_strip_bins = 3 #One per VFAT
x_time_res_xAvgFloat = np.empty([n_strip_bins*len(eta_list)])
y_time_res_xAvgFloat = np.empty([n_strip_bins*len(eta_list)])
y_time_res_err_xAvgFloat = np.empty([n_strip_bins*len(eta_list)])

x_time_res_xAvgInt = np.empty([n_strip_bins*len(eta_list)])
y_time_res_xAvgInt = np.empty([n_strip_bins*len(eta_list)])
y_time_res_err_xAvgInt = np.empty([n_strip_bins*len(eta_list)])

x_time_res_xFirst = np.empty([n_strip_bins*len(eta_list)])
y_time_res_xFirst = np.empty([n_strip_bins*len(eta_list)])
y_time_res_err_xFirst = np.empty([n_strip_bins*len(eta_list)])

x_time_res_xLast = np.empty([n_strip_bins*len(eta_list)])
y_time_res_xLast = np.empty([n_strip_bins*len(eta_list)])
y_time_res_err_xLast = np.empty([n_strip_bins*len(eta_list)])

fig_BX_eta3_VFAT2, axs_BX_eta3_VFAT2 = plt.subplots(4, figsize=(20,20))


for eta_num in eta_list:
    print("Starting Eta ", eta_num)

    #Make all segment combinations between ch0 and ch3
    ch0_strip = strip[(ch == 0) & (eta == eta_num)]
    ch1_strip = strip[(ch == 1) & (eta == eta_num)]
    ch2_strip = strip[(ch == 2) & (eta == eta_num)]
    ch3_strip = strip[(ch == 3) & (eta == eta_num)]

    ch0_z = ch[(ch == 0) & (eta == eta_num)]*36
    ch1_z = ch[(ch == 1) & (eta == eta_num)]*36
    ch2_z = ch[(ch == 2) & (eta == eta_num)]*36
    ch3_z = ch[(ch == 3) & (eta == eta_num)]*36

    ch0_error = strip_error[(ch == 0) & (eta == eta_num)]
    ch1_error = strip_error[(ch == 1) & (eta == eta_num)]
    ch2_error = strip_error[(ch == 2) & (eta == eta_num)]
    ch3_error = strip_error[(ch == 3) & (eta == eta_num)]

    ch0_BX = BX_first[(ch == 0) & (eta == eta_num)]
    ch1_BX = BX_first[(ch == 1) & (eta == eta_num)]
    ch2_BX = BX_first[(ch == 2) & (eta == eta_num)]
    ch3_BX = BX_first[(ch == 3) & (eta == eta_num)]


    combinations = ak.cartesian([ch0_strip, ch1_strip, ch2_strip, ch3_strip])
    one, two, three, four = ak.unzip(combinations)
    one = ak.singletons(one)
    two = ak.singletons(two)
    three = ak.singletons(three)
    four = ak.singletons(four)
    strip_combinations = ak.concatenate([one, two, three, four], axis=2)

    zpos = ak.cartesian([ch0_z, ch1_z, ch2_z, ch3_z])
    zone, ztwo, zthree, zfour = ak.unzip(zpos)
    zone = ak.singletons(zone)
    ztwo = ak.singletons(ztwo)
    zthree = ak.singletons(zthree)
    zfour = ak.singletons(zfour)
    z_combinations = ak.concatenate([zone, ztwo, zthree, zfour], axis=2)

    errors = ak.cartesian([ch0_error, ch1_error, ch2_error, ch3_error])
    errorone, errortwo, errorthree, errorfour = ak.unzip(errors)
    errorone = ak.singletons(errorone)
    errortwo = ak.singletons(errortwo)
    errorthree = ak.singletons(errorthree)
    errorfour = ak.singletons(errorfour)
    error_combinations = ak.concatenate([errorone, errortwo, errorthree, errorfour], axis=2)

    BXs = ak.cartesian([ch0_BX, ch1_BX, ch2_BX, ch3_BX])
    BXone, BXtwo, BXthree, BXfour = ak.unzip(BXs)
    BXone = ak.singletons(BXone)
    BXtwo = ak.singletons(BXtwo)
    BXthree = ak.singletons(BXthree)
    BXfour = ak.singletons(BXfour)
    BX_combinations = ak.concatenate([BXone, BXtwo, BXthree, BXfour], axis=2)

    fit = ak.linear_fit(z_combinations, strip_combinations, axis=2)
    slopes = fit.slope
    intercepts = fit.intercept

    #Calculate chi2 (measure - expected)^2 / (sigma^2)
    #Cluster strip position error is clusterSize / sqrt(12)
    expected_strips = slopes*z_combinations + intercepts

    residuals = strip_combinations - expected_strips
    chi2 = ((residuals)**2) / (error_combinations**2)
    #Need to divide sum by (nClusters-1)
    chi2_sum = ak.sum(chi2, axis=2)/(4-1)

    #Check how the results look
    #print(*strip_combinations[0])
    #print(*error_combinations[0])
    #print(*expected_strips[0])
    #print(*residuals[0])
    #print(*chi2[0])
    #print(*chi2_sum[0])

    best_segments_arg = ak.argmin(chi2_sum, axis=1, keepdims=True) #keepdims is required due to using axis=1

    segment_strips = strip_combinations[best_segments_arg]
    segment_residuals = residuals[best_segments_arg]
    segment_BX_firsts = BX_combinations[best_segments_arg]


    ch0_strips = segment_strips[:,:,0]
    ch0_residuals = segment_residuals[:,:,0]
    ch0_BXs = segment_BX_firsts[:,:,0]

    ch0_strips_numpy = ak.to_numpy(ak.flatten(ch0_strips[~ak.is_none(ch0_strips, axis=1)], axis=1))
    ch0_residuals_numpy = ak.to_numpy(ak.flatten(ch0_residuals[~ak.is_none(ch0_residuals, axis=1)], axis=1))
    ch0_BX_numpy = ak.to_numpy(ak.flatten(ch0_BXs[~ak.is_none(ch0_BXs, axis=1)], axis=1))


    ch1_strips = segment_strips[:,:,1]
    ch1_residuals = segment_residuals[:,:,1]
    ch1_BXs = segment_BX_firsts[:,:,1]

    ch1_strips_numpy = ak.to_numpy(ak.flatten(ch1_strips[~ak.is_none(ch1_strips, axis=1)], axis=1))
    ch1_residuals_numpy = ak.to_numpy(ak.flatten(ch1_residuals[~ak.is_none(ch1_residuals, axis=1)], axis=1))
    ch1_BX_numpy = ak.to_numpy(ak.flatten(ch1_BXs[~ak.is_none(ch1_BXs, axis=1)], axis=1))


    ch2_strips = segment_strips[:,:,2]
    ch2_residuals = segment_residuals[:,:,2]
    ch2_BXs = segment_BX_firsts[:,:,2]
    """
    #Need to correct the VFATs in Ch2/3, move BX down
    ch2_BXs = ak.where(
            ((ch2_strips < 255) & ((eta_num == 1) | (eta_num == 4))),
                ch2_BXs - 1.0,
                ch2_BXs
        )
    """
    ch2_BXs = ak.where(
            ((ch2_strips < 255) & ((eta_num == 1) | (eta_num == 4))) == 0,
                ch2_BXs + 1.0,
                ch2_BXs
        )

    ch2_strips_numpy = ak.to_numpy(ak.flatten(ch2_strips[~ak.is_none(ch2_strips, axis=1)], axis=1))
    ch2_residuals_numpy = ak.to_numpy(ak.flatten(ch2_residuals[~ak.is_none(ch2_residuals, axis=1)], axis=1))
    ch2_BX_numpy = ak.to_numpy(ak.flatten(ch2_BXs[~ak.is_none(ch2_BXs, axis=1)], axis=1))


    ch3_strips = segment_strips[:,:,3]
    ch3_residuals = segment_residuals[:,:,3]
    ch3_BXs = segment_BX_firsts[:,:,3]
    """
    #Need to correct the VFATs in Ch2/3, move BX down
    ch3_BXs = ak.where(
            (eta_num == 1) | ((eta_num == 2) & (ch3_strips < 127)) | ((eta_num == 3) & (ch3_strips > 127) & (ch3_strips < 255)),
                ch3_BXs - 1.0,
                ch3_BXs
        )
    """
    ch3_BXs = ak.where(
            ((eta_num == 1) | ((eta_num == 2) & (ch3_strips < 127)) | ((eta_num == 3) & (ch3_strips > 127) & (ch3_strips < 255))) == 0,
                ch3_BXs + 1.0,
                ch3_BXs
        )

    ch3_strips_numpy = ak.to_numpy(ak.flatten(ch3_strips[~ak.is_none(ch3_strips, axis=1)], axis=1))
    ch3_residuals_numpy = ak.to_numpy(ak.flatten(ch3_residuals[~ak.is_none(ch3_residuals, axis=1)], axis=1))
    ch3_BX_numpy = ak.to_numpy(ak.flatten(ch3_BXs[~ak.is_none(ch3_BXs, axis=1)], axis=1))



    tmp_occupancy = axs_occupancy[0, eta_num-1].hist2d(ch0_strips_numpy, ch0_residuals_numpy, bins=residual_bins, range=residual_axis_range)
    fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=axs_occupancy[eta_num-1, 0])
    axs_occupancy[0, eta_num-1].set_xlabel("Cluster Strip")
    axs_occupancy[0, eta_num-1].set_ylabel("Residual")
    axs_occupancy[0, eta_num-1].set_title("Residual Ch0 Eta{}".format(eta_num))


    tmp_occupancy = axs_occupancy[1, eta_num-1].hist2d(ch1_strips_numpy, ch1_residuals_numpy, bins=residual_bins, range=residual_axis_range)
    fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=axs_occupancy[eta_num-1, 1])
    axs_occupancy[1, eta_num-1].set_xlabel("Cluster Strip")
    axs_occupancy[1, eta_num-1].set_ylabel("Residual")
    axs_occupancy[1, eta_num-1].set_title("Residual Ch1 Eta{}".format(eta_num))


    tmp_occupancy = axs_occupancy[2, eta_num-1].hist2d(ch2_strips_numpy, ch2_residuals_numpy, bins=residual_bins, range=residual_axis_range)
    fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=axs_occupancy[eta_num-1, 2])
    axs_occupancy[2, eta_num-1].set_xlabel("Cluster Strip")
    axs_occupancy[2, eta_num-1].set_ylabel("Residual")
    axs_occupancy[2, eta_num-1].set_title("Residual Ch2 Eta{}".format(eta_num))


    tmp_occupancy = axs_occupancy[3, eta_num-1].hist2d(ch3_strips_numpy, ch3_residuals_numpy, bins=residual_bins, range=residual_axis_range)
    fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=axs_occupancy[eta_num-1, 3])
    axs_occupancy[3, eta_num-1].set_xlabel("Cluster Strip")
    axs_occupancy[3, eta_num-1].set_ylabel("Residual")
    axs_occupancy[3, eta_num-1].set_title("Residual Ch3 Eta{}".format(eta_num))







    tmp_occupancy = bx_axs_occupancy[0, eta_num-1].hist2d(ch0_strips_numpy, ch0_BX_numpy, bins=bx_bins, range=bx_axis_range)
    bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=bx_axs_occupancy[eta_num-1, 0])
    bx_axs_occupancy[0, eta_num-1].set_xlabel("Cluster Strip")
    bx_axs_occupancy[0, eta_num-1].set_ylabel("BX")
    bx_axs_occupancy[0, eta_num-1].set_title("BX Ch0 Eta{}".format(eta_num))


    tmp_occupancy = bx_axs_occupancy[1, eta_num-1].hist2d(ch1_strips_numpy, ch1_BX_numpy, bins=bx_bins, range=bx_axis_range)
    bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=bx_axs_occupancy[eta_num-1, 1])
    bx_axs_occupancy[1, eta_num-1].set_xlabel("Cluster Strip")
    bx_axs_occupancy[1, eta_num-1].set_ylabel("BX")
    bx_axs_occupancy[1, eta_num-1].set_title("BX Ch1 Eta{}".format(eta_num))


    tmp_occupancy = bx_axs_occupancy[2, eta_num-1].hist2d(ch2_strips_numpy, ch2_BX_numpy, bins=bx_bins, range=bx_axis_range)
    bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=bx_axs_occupancy[eta_num-1, 2])
    bx_axs_occupancy[2, eta_num-1].set_xlabel("Cluster Strip")
    bx_axs_occupancy[2, eta_num-1].set_ylabel("BX")
    bx_axs_occupancy[2, eta_num-1].set_title("BX Ch2 Eta{}".format(eta_num))


    tmp_occupancy = bx_axs_occupancy[3, eta_num-1].hist2d(ch3_strips_numpy, ch3_BX_numpy, bins=bx_bins, range=bx_axis_range)
    bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=bx_axs_occupancy[eta_num-1, 3])
    bx_axs_occupancy[3, eta_num-1].set_xlabel("Cluster Strip")
    bx_axs_occupancy[3, eta_num-1].set_ylabel("BX")
    bx_axs_occupancy[3, eta_num-1].set_title("BX Ch3 Eta{}".format(eta_num))



    mean_strip = ak.mean(segment_strips, axis=2)
    mean_BX = ak.mean(segment_BX_firsts, axis=2)
    first_BX = ak.min(segment_BX_firsts, axis=2)
    last_BX = ak.max(segment_BX_firsts, axis=2)

    segment_strip = ak.to_numpy(ak.flatten(mean_strip[~ak.is_none(mean_strip, axis=1)]))
    segment_BX_avg = ak.to_numpy(ak.flatten(mean_BX[~ak.is_none(mean_BX, axis=1)]))
    segment_BX_first = ak.to_numpy(ak.flatten(first_BX[~ak.is_none(first_BX, axis=1)]))
    segment_BX_last = ak.to_numpy(ak.flatten(last_BX[~ak.is_none(last_BX, axis=1)]))
    segment_BX_avg_int = np.round(segment_BX_avg, 0)

    tmp_occupancy = clustAvgFl_bx_axs_occupancy[eta_num-1].hist2d(segment_strip, segment_BX_avg, bins=clust_bx_bins, range=clust_bx_axis_range)
    clustAvgFl_bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=clustAvgFl_bx_axs_occupancy[eta_num-1])
    clustAvgFl_bx_axs_occupancy[eta_num-1].set_xlabel("Cluster Strip")
    clustAvgFl_bx_axs_occupancy[eta_num-1].set_ylabel("Average BX (Float)")
    clustAvgFl_bx_axs_occupancy[eta_num-1].set_title("Average Segment BX (Float) Eta{}".format(eta_num))

    tmp_occupancy = clustAvgInt_bx_axs_occupancy[eta_num-1].hist2d(segment_strip, segment_BX_avg_int, bins=clust_bx_bins, range=clust_bx_axis_range)
    clustAvgInt_bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=clustAvgInt_bx_axs_occupancy[eta_num-1])
    clustAvgInt_bx_axs_occupancy[eta_num-1].set_xlabel("Cluster Strip")
    clustAvgInt_bx_axs_occupancy[eta_num-1].set_ylabel("Average BX (Int)")
    clustAvgInt_bx_axs_occupancy[eta_num-1].set_title("Average Segment BX (Int) Eta{}".format(eta_num))

    tmp_occupancy = clustFirst_bx_axs_occupancy[eta_num-1].hist2d(segment_strip, segment_BX_first, bins=clust_bx_bins, range=clust_bx_axis_range)
    clustFirst_bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=clustFirst_bx_axs_occupancy[eta_num-1])
    clustFirst_bx_axs_occupancy[eta_num-1].set_xlabel("Cluster Strip")
    clustFirst_bx_axs_occupancy[eta_num-1].set_ylabel("First BX")
    clustFirst_bx_axs_occupancy[eta_num-1].set_title("First Segment BX Eta{}".format(eta_num))

    tmp_occupancy = clustLast_bx_axs_occupancy[eta_num-1].hist2d(segment_strip, segment_BX_last, bins=clust_bx_bins, range=clust_bx_axis_range)
    clustLast_bx_fig_occupancy.colorbar(tmp_occupancy[3], label="Number of Entries", ax=clustLast_bx_axs_occupancy[eta_num-1])
    clustLast_bx_axs_occupancy[eta_num-1].set_xlabel("Cluster Strip")
    clustLast_bx_axs_occupancy[eta_num-1].set_ylabel("Last BX")
    clustLast_bx_axs_occupancy[eta_num-1].set_title("Last Segment BX Eta{}".format(eta_num))


    #The ideal BX for each eta over strip bins

    for strip_bin_num in range(n_strip_bins):
        strip_start = (strip_bin_num)*(128*3)/n_strip_bins
        strip_end = (strip_bin_num+1)*(128*3)/n_strip_bins
        #Look at all digiEvents on eta while in awkward

        xAvgFloat = segment_BX_avg
        xAvgInt = segment_BX_avg_int
        xFirst = segment_BX_first
        xLast = segment_BX_last
        y = segment_strip
        mask = (y >= strip_start) & (y <= strip_end)

        xAvgFloat1 = xAvgFloat[mask]
        xAvgInt1 = xAvgInt[mask]
        xFirst1 = xFirst[mask]
        xLast1 = xLast[mask]
        y1 = y[mask]

        (mu, sigma) = norm.fit(xAvgFloat1)
        mu = np.nan_to_num(mu, 0)
        sigma = np.nan_to_num(sigma, 0)
        x_time_res_xAvgFloat[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
        y_time_res_xAvgFloat[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
        y_time_res_err_xAvgFloat[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma

        (mu, sigma) = norm.fit(xAvgInt1)
        mu = np.nan_to_num(mu, 0)
        sigma = np.nan_to_num(sigma, 0)
        x_time_res_xAvgInt[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
        y_time_res_xAvgInt[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
        y_time_res_err_xAvgInt[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma

        (mu, sigma) = norm.fit(xFirst)
        mu = np.nan_to_num(mu, 0)
        sigma = np.nan_to_num(sigma, 0)
        x_time_res_xFirst[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
        y_time_res_xFirst[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
        y_time_res_err_xFirst[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma

        (mu, sigma) = norm.fit(xLast)
        mu = np.nan_to_num(mu, 0)
        sigma = np.nan_to_num(sigma, 0)
        x_time_res_xLast[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
        y_time_res_xLast[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
        y_time_res_err_xLast[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma


    if eta_num == 3:
        x = ch0_BX_numpy
        mask = (ch0_strips_numpy > 127) & (ch0_strips_numpy < 255)
        x = x[mask]
        tmp_eta3_stripsVFAT2 = axs_BX_eta3_VFAT2[0].hist(x, bins=7, range=(0.0,7.0))
        axs_BX_eta3_VFAT2[0].set_xlabel("Cluster BX")
        axs_BX_eta3_VFAT2[0].set_ylabel("Entries")
        axs_BX_eta3_VFAT2[0].set_title("BX Ch0 Eta3 VFAT2")
        fig_BX_eta3_VFAT2.text(0.8, 0.25*((3-0)+1)-0.05, "Sigma: {}".format(round(norm.fit(x)[1],3)), fontsize=24)

        x = ch1_BX_numpy
        mask = (ch1_strips_numpy > 127) & (ch1_strips_numpy < 255)
        x = x[mask]
        tmp_eta3_stripsVFAT2 = axs_BX_eta3_VFAT2[1].hist(x, bins=7, range=(0.0,7.0))
        axs_BX_eta3_VFAT2[1].set_xlabel("Cluster BX")
        axs_BX_eta3_VFAT2[1].set_ylabel("Entries")
        axs_BX_eta3_VFAT2[1].set_title("BX Ch1 Eta3 VFAT2")
        fig_BX_eta3_VFAT2.text(0.8, 0.25*((3-1)+1)-0.05, "Sigma: {}".format(round(norm.fit(x)[1],3)), fontsize=24)

        x = ch2_BX_numpy
        mask = (ch2_strips_numpy > 127) & (ch2_strips_numpy < 255)
        x = x[mask]
        tmp_eta3_stripsVFAT2 = axs_BX_eta3_VFAT2[2].hist(x, bins=7, range=(0.0,7.0))
        axs_BX_eta3_VFAT2[2].set_xlabel("Cluster BX")
        axs_BX_eta3_VFAT2[2].set_ylabel("Entries")
        axs_BX_eta3_VFAT2[2].set_title("BX Ch2 Eta3 VFAT2")
        fig_BX_eta3_VFAT2.text(0.8, 0.25*((3-2)+1)-0.05, "Sigma: {}".format(round(norm.fit(x)[1],3)), fontsize=24)

        x = ch3_BX_numpy
        mask = (ch3_strips_numpy > 127) & (ch3_strips_numpy < 255)
        x = x[mask]
        tmp_eta3_stripsVFAT2 = axs_BX_eta3_VFAT2[3].hist(x, bins=7, range=(0.0,7.0))
        axs_BX_eta3_VFAT2[3].set_xlabel("Cluster BX")
        axs_BX_eta3_VFAT2[3].set_ylabel("Entries")
        axs_BX_eta3_VFAT2[3].set_title("BX Ch3 Eta3 VFAT2")
        fig_BX_eta3_VFAT2.text(0.8, 0.25*((3-3)+1)-0.05, "Sigma: {}".format(round(norm.fit(x)[1],3)), fontsize=24)


fig_occupancy.tight_layout()
fig_occupancy.savefig(plot_prefix+"clusterresidual_segments.png")
plt.close(fig_occupancy)


bx_fig_occupancy.tight_layout()
bx_fig_occupancy.savefig(plot_prefix+"clusterBX_segments.png")
plt.close(bx_fig_occupancy)


clustAvgFl_bx_fig_occupancy.tight_layout()
clustAvgFl_bx_fig_occupancy.savefig(plot_prefix+"averageFloatBX_segments.png")
plt.close(bx_fig_occupancy)

clustAvgInt_bx_fig_occupancy.tight_layout()
clustAvgInt_bx_fig_occupancy.savefig(plot_prefix+"averageIntBX_segments.png")
plt.close(bx_fig_occupancy)

clustFirst_bx_fig_occupancy.tight_layout()
clustFirst_bx_fig_occupancy.savefig(plot_prefix+"firstBX_segments.png")
plt.close(bx_fig_occupancy)

clustLast_bx_fig_occupancy.tight_layout()
clustLast_bx_fig_occupancy.savefig(plot_prefix+"lastBX_segments.png")
plt.close(bx_fig_occupancy)




fig_timeres, axs_timeres = plt.subplots(1, figsize=(20,20))

fig_timeres_hist, axs_timeres_hist = plt.subplots(1, figsize=(20,20))

tmp_time_res = axs_timeres.errorbar(x_time_res_xAvgFloat, y_time_res_xAvgFloat, yerr=y_time_res_err_xAvgFloat)
axs_timeres.set_xlabel("VFAT")
axs_timeres.set_ylabel("Mean")
axs_timeres.set_title("Segment Time Resolution Avg Float")
axs_timeres.set_ylim(1,5)


tmp_time_res_hist = axs_timeres_hist.hist(y_time_res_err_xAvgFloat, bins=20, range=(0.3,0.8))
axs_timeres_hist.set_xlabel("Sigma")
axs_timeres_hist.set_ylabel("Entries")
axs_timeres_hist.set_title("Time Resolution Avg Float")
(mu, sigma) = norm.fit(y_time_res_err_xAvgFloat)
mu = np.nan_to_num(mu, 0)
sigma = np.nan_to_num(sigma, 0)
fig_timeres_hist.text(0.8, 0.8, "Mean: {}".format(round(mu,3)), fontsize=24)

fig_timeres.tight_layout()
fig_timeres.savefig(plot_prefix+"time_res_avgfloat.pdf")
plt.close(fig_timeres)

fig_timeres_hist.tight_layout()
fig_timeres_hist.savefig(plot_prefix+"time_res_hist_avgfloat.pdf")
plt.close(fig_timeres_hist)



fig_timeres, axs_timeres = plt.subplots(1, figsize=(20,20))

fig_timeres_hist, axs_timeres_hist = plt.subplots(1, figsize=(20,20))

tmp_time_res = axs_timeres.errorbar(x_time_res_xAvgInt, y_time_res_xAvgInt, yerr=y_time_res_err_xAvgInt)
axs_timeres.set_xlabel("VFAT")
axs_timeres.set_ylabel("Mean")
axs_timeres.set_title("Segment Time Resolution Avg Int")
axs_timeres.set_ylim(1,5)


tmp_time_res_hist = axs_timeres_hist.hist(y_time_res_err_xAvgInt, bins=20, range=(0.3,0.8))
axs_timeres_hist.set_xlabel("Sigma")
axs_timeres_hist.set_ylabel("Entries")
axs_timeres_hist.set_title("Time Resolution Avg Int")
(mu, sigma) = norm.fit(y_time_res_err_xAvgInt)
mu = np.nan_to_num(mu, 0)
sigma = np.nan_to_num(sigma, 0)
fig_timeres_hist.text(0.8, 0.8, "Mean: {}".format(round(mu,3)), fontsize=24)

fig_timeres.tight_layout()
fig_timeres.savefig(plot_prefix+"time_res_avgint.pdf")
plt.close(fig_timeres)

fig_timeres_hist.tight_layout()
fig_timeres_hist.savefig(plot_prefix+"time_res_hist_avgint.pdf")
plt.close(fig_timeres_hist)





fig_timeres, axs_timeres = plt.subplots(1, figsize=(20,20))

fig_timeres_hist, axs_timeres_hist = plt.subplots(1, figsize=(20,20))

tmp_time_res = axs_timeres.errorbar(x_time_res_xFirst, y_time_res_xFirst, yerr=y_time_res_err_xFirst)
axs_timeres.set_xlabel("VFAT")
axs_timeres.set_ylabel("Mean")
axs_timeres.set_title("Segment Time Resolution First")
axs_timeres.set_ylim(1,5)


tmp_time_res_hist = axs_timeres_hist.hist(y_time_res_err_xFirst, bins=20, range=(0.3,0.8))
axs_timeres_hist.set_xlabel("Sigma")
axs_timeres_hist.set_ylabel("Entries")
axs_timeres_hist.set_title("Time Resolution First")
(mu, sigma) = norm.fit(y_time_res_err_xFirst)
mu = np.nan_to_num(mu, 0)
sigma = np.nan_to_num(sigma, 0)
fig_timeres_hist.text(0.8, 0.8, "Mean: {}".format(round(mu,3)), fontsize=24)

fig_timeres.tight_layout()
fig_timeres.savefig(plot_prefix+"time_res_first.pdf")
plt.close(fig_timeres)

fig_timeres_hist.tight_layout()
fig_timeres_hist.savefig(plot_prefix+"time_res_hist_first.pdf")
plt.close(fig_timeres_hist)





fig_timeres, axs_timeres = plt.subplots(1, figsize=(20,20))

fig_timeres_hist, axs_timeres_hist = plt.subplots(1, figsize=(20,20))

tmp_time_res = axs_timeres.errorbar(x_time_res_xLast, y_time_res_xLast, yerr=y_time_res_err_xLast)
axs_timeres.set_xlabel("VFAT")
axs_timeres.set_ylabel("Mean")
axs_timeres.set_title("Segment Time Resolution Last")
axs_timeres.set_ylim(1,5)


tmp_time_res_hist = axs_timeres_hist.hist(y_time_res_err_xLast, bins=20, range=(0.3,0.8))
axs_timeres_hist.set_xlabel("Sigma")
axs_timeres_hist.set_ylabel("Entries")
axs_timeres_hist.set_title("Time Resolution Last")
(mu, sigma) = norm.fit(y_time_res_err_xLast)
mu = np.nan_to_num(mu, 0)
sigma = np.nan_to_num(sigma, 0)
fig_timeres_hist.text(0.8, 0.8, "Mean: {}".format(round(mu,3)), fontsize=24)

fig_timeres.tight_layout()
fig_timeres.savefig(plot_prefix+"time_res_last.pdf")
plt.close(fig_timeres)

fig_timeres_hist.tight_layout()
fig_timeres_hist.savefig(plot_prefix+"time_res_hist_last.pdf")
plt.close(fig_timeres_hist)



fig_BX_eta3_VFAT2.tight_layout()
fig_BX_eta3_VFAT2.savefig(plot_prefix+"eta3_VFAT2_clustBX.pdf")
plt.close(fig_BX_eta3_VFAT2)
