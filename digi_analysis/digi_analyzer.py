import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm

#Script to take the unpacked digi file and create plots on the digiBX timing


#Folder to put the plots in
#plot_prefix = "plots/run386/"
plot_prefix = "plots/run388/"

#Input file name
#fname = "../me0_multibx_digi_run386.root"
fname = "../me0_multibx_digi_run388.root"

#New data format 28Nov23
#orbitNumber     = 45905
#bunchCounter    = 1820
#eventCounter    = 32
#runParameter    = 455
#pulse_stretch   = 0
#rawSlot         = (vector<int>*)0x600001914920
#rawOH           = (vector<int>*)0x600001914940
#rawVFAT         = (vector<int>*)0x600001914e20
#rawChannel      = (vector<int>*)0x600001914c20
#digiBX          = (vector<int>*)0x600001915120
#digiStripChamber = (vector<int>*)0x600001914980
#digiStripEta    = (vector<int>*)0x600001915160
#digiStrip       = (vector<int>*)0x600001914900
#digiStripCharge = (vector<int>*)0x6000019148e0
#digiStripTime   = (vector<int>*)0x6000019148c0
#digiPadChamber  = (vector<int>*)0x6000019148a0
#digiPadX        = (vector<int>*)0x600001914880
#digiPadY        = (vector<int>*)0x600001914860
#digiPadCharge   = (vector<int>*)0x600001914840
#digiPadTime     = (vector<int>*)0x600001914820



uproot_file = uproot.open(fname)
tree = uproot_file['outputtree']
events = tree.arrays()


ch_list = [0,1,2,3]
eta_list = [1,2,3,4]
vfat_list = [4,5,6,7,12,13,14,15,20,21,22,23]
n_strip_bins = int(128*3 / 8) #Number of bins per eta partition
n_stips_per_bin = 128*3/n_strip_bins

for ch_num in ch_list:
    events = ak.with_field(events, events.rawOH[events.digiStripChamber == ch_num], 'OH_ch{}'.format(ch_num))
    events = ak.with_field(events, events.rawVFAT[events.digiStripChamber == ch_num], 'VFAT_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiBX[events.digiStripChamber == ch_num], 'digiBX_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiStripChamber[events.digiStripChamber == ch_num], 'digiChamber_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiStripEta[events.digiStripChamber == ch_num], 'digiEta_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiStrip[events.digiStripChamber == ch_num], 'digiStrip_ch{}'.format(ch_num))
print("Created event specific lists! ", events.fields)

#Figure for the 2D BX Occupancy
fig_occu, axs_occu = plt.subplots(4, 4, figsize=(20,20))
#Figure for the 2D Digi Multiplicity
fig_mult, axs_mult = plt.subplots(4, 4, figsize=(20,20))

#Figure for the time resolution
fig_timeres, axs_timeres = plt.subplots(4, figsize=(20,20))
fig_timeres_hist, axs_timeres_hist = plt.subplots(4, figsize=(20,20))

#Figure for the time resolution split by eta partitions
fig_timeres_hist_by_eta, axs_timeres_hist_by_eta = plt.subplots(4, 4, figsize=(20,20))

#Figures for the 1D BX distributions per chamber per vfat
figure_list = [plt.subplots(3, 4, figsize=(20,20)), plt.subplots(3, 4, figsize=(20,20)), plt.subplots(3, 4, figsize=(20,20)), plt.subplots(3, 4, figsize=(20,20))]

if not os.path.exists(plot_prefix):
    os.makedirs(plot_prefix)

for ch_num in ch_list:
    print("At chamber ", ch_num)
    x_time_res = np.empty([n_strip_bins*len(eta_list)])
    y_time_res = np.empty([n_strip_bins*len(eta_list)])
    y_time_res_err = np.empty([n_strip_bins*len(eta_list)])

    fig_bx_hist_ch, axs_bx_hist_ch = figure_list[ch_num-1]
    #BX 1D Hists
    for i, vfat in enumerate(vfat_list):
        mask = (events['VFAT_ch{}'.format(ch_num)] == vfat) & (events['digiBX_ch{}'.format(ch_num)] < 7) & (events['digiBX_ch{}'.format(ch_num)] > 1)
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        (mu, sigma) = norm.fit(x)
        tmp_bx_hist = axs_bx_hist_ch[int(vfat/8), vfat%4].hist(x, bins=(5), range=(1.5,6.5))
        axs_bx_hist_ch[int(vfat/8), vfat%4].set_xlabel("BX")
        axs_bx_hist_ch[int(vfat/8), vfat%4].set_title("Digi BX Ch{} VFAT{}".format(ch_num, vfat))
        axs_bx_hist_ch[int(vfat/8), vfat%4].text(0.5, 0.9, "StdDev: {}".format(round(sigma,3)), fontsize=18, transform=axs_bx_hist_ch[int(vfat/8), vfat%4].transAxes)
    fig_bx_hist_ch.tight_layout()
    fig_bx_hist_ch.savefig(plot_prefix+"bx_1d_ch{}_vfats.pdf".format(ch_num))

    for eta_num in eta_list:
        #Digi Occupancy
        mask = events['digiEta_ch{}'.format(ch_num)] == eta_num
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        tmp_occu = axs_occu[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        fig_occu.colorbar(tmp_occu[3], label="Number of Entries", ax=axs_occu[eta_num-1, ch_num])
        axs_occu[eta_num-1, ch_num].set_xlabel("BX")
        axs_occu[eta_num-1, ch_num].set_ylabel("Digi Strip")
        axs_occu[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))



        #The ideal BX for each eta over strip bins
        for strip_bin_num in range(n_strip_bins):
            strip_start = (strip_bin_num)*(128*3)/n_strip_bins
            strip_end = (strip_bin_num+1)*(128*3)/n_strip_bins
            #Look at all digiEvents on eta while in awkward
            mask = events['digiEta_ch{}'.format(ch_num)] == eta_num

            x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
            y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

            #Only look at BX less than or equal to 7 and strip within bins in the flat numpy
            mask = (x < 7) & (x > 1) & (y >= strip_start) & (y <= strip_end)
            x1 = x[mask]
            y1 = y[mask]

            (mu, sigma) = norm.fit(x1)
            x_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
            y_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
            y_time_res_err[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma



        #Digi Multiplicity
        BX_for_mult = []
        mult_for_mult = []
        for BX in range(16):
            x = ak.to_numpy(ak.count(events['digiStrip_ch{}'.format(ch_num)][(events['digiEta_ch{}'.format(ch_num)] == eta_num) & (events['digiBX_ch{}'.format(ch_num)] == BX)], axis=1))
            BX_values = np.full_like(x, BX)

            BX_for_mult.append(BX_values)
            mult_for_mult.append(x)

        BX_for_mult = np.array(BX_for_mult).flatten()
        mult_for_mult = np.array(mult_for_mult).flatten()

        x = BX_for_mult
        y = mult_for_mult

        tmp_mult = axs_mult[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        fig_mult.colorbar(tmp_mult[3], label="Number of Entries", ax=axs_mult[eta_num-1, ch_num])
        axs_mult[eta_num-1, ch_num].set_xlabel("BX")
        axs_mult[eta_num-1, ch_num].set_ylabel("Digi Multiplicity")
        axs_mult[eta_num-1, ch_num].set_title("Digi Multiplicity Ch{} Eta{}".format(ch_num, eta_num))


        y_time_res_err_eta = y_time_res_err[(eta_num-1)*n_strip_bins:(eta_num)*n_strip_bins]
        tmp_time_res_hist_by_eta = axs_timeres_hist_by_eta[eta_num-1, ch_num].hist(y_time_res_err_eta, bins=20, range=(0.5,1.0))
        axs_timeres_hist_by_eta[eta_num-1, ch_num].set_xlabel("BX Sigma")
        axs_timeres_hist_by_eta[eta_num-1, ch_num].set_ylabel("Entries")
        axs_timeres_hist_by_eta[eta_num-1, ch_num].set_title("Time Resolution Ch{} Eta{}".format(ch_num, eta_num))
        y_time_res_err_eta_nonan = y_time_res_err_eta[~np.isnan(y_time_res_err_eta)]
        fig_mean_by_eta = round(norm.fit(y_time_res_err_eta_nonan)[0],3)
        axs_timeres_hist_by_eta[eta_num-1, ch_num].text(0.5, 0.8, "Mean: {}".format(fig_mean_by_eta), fontsize=18, transform=axs_timeres_hist_by_eta[eta_num-1, ch_num].transAxes)




    tmp_time_res = axs_timeres[ch_num].errorbar(x_time_res, y_time_res, yerr=y_time_res_err)
    axs_timeres[ch_num].set_xlabel("Strip Bin (x{})".format(n_stips_per_bin))
    axs_timeres[ch_num].set_ylabel("Mean BX")
    axs_timeres[ch_num].set_title("Time Resolution Ch{}".format(ch_num, eta_num))
    axs_timeres[ch_num].set_ylim(1,5)

    tmp_time_res_hist = axs_timeres_hist[ch_num].hist(y_time_res_err, bins=50, range=(0.5,1.0))
    axs_timeres_hist[ch_num].set_xlabel("BX Sigma")
    axs_timeres_hist[ch_num].set_ylabel("Entries")
    axs_timeres_hist[ch_num].set_title("Time Resolution Ch{}".format(ch_num, eta_num))
    y_time_res_err_nonan = y_time_res_err[~np.isnan(y_time_res_err)]
    fig_mean = round(norm.fit(y_time_res_err_nonan)[0],3)
    fig_timeres_hist.text(0.8, 0.25*((3-ch_num)+1)-0.05, "Mean: {}".format(fig_mean), fontsize=24)
    #axs_timeres_hist[ch_num].set_xlim(0,1.5)


fig_occu.tight_layout()
fig_occu.savefig(plot_prefix+"occupancy.pdf".format(ch_num, eta_num))

fig_mult.tight_layout()
fig_mult.savefig(plot_prefix+"multiplicity.pdf".format(ch_num, eta_num))

fig_timeres.tight_layout()
fig_timeres.savefig(plot_prefix+"time_res.pdf")

fig_timeres_hist.tight_layout()
fig_timeres_hist.savefig(plot_prefix+"time_res_hist.pdf")

fig_timeres_hist_by_eta.tight_layout()
fig_timeres_hist_by_eta.savefig(plot_prefix+"time_res_hist_by_eta.pdf")
