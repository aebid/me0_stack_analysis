import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

plot_prefix = "plots/run154/"
plot_prefix = "plots/run146/"

#fname = "../input_files/digi/00000154.root"
fname = "../input_files/digi/00000146.root"

#Data looks like
#orbitNumber     = 15754
#bunchCounter    = 611
#eventCounter    = 16
#runParameter    = 455
#pulse_stretch   = 0
#slot            = (vector<int>*)0x60000304e640
#OH              = (vector<int>*)0x600003053460
#VFAT            = (vector<int>*)0x60000304dfa0
#CH              = (vector<int>*)0x6000030548e0
#digiBX          = (vector<int>*)0x600003041c40
#digiChamber     = (vector<int>*)0x600003053740
#digiEta         = (vector<int>*)0x600003054620
#digiDirection   = (vector<int>*)0x600003041420
#digiStrip       = (vector<int>*)0x600003041ae0


uproot_file = uproot.open(fname)
tree = uproot_file['outputtree']
events = tree.arrays()


ch_list = [0,1,2,3]
eta_list = [1,2,3,4]

for ch_num in ch_list:
    events = ak.with_field(events, events.OH[events.digiChamber == ch_num], 'OH_ch{}'.format(ch_num))
    events = ak.with_field(events, events.VFAT[events.digiChamber == ch_num], 'VFAT_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiBX[events.digiChamber == ch_num], 'digiBX_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiChamber[events.digiChamber == ch_num], 'digiChamber_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiEta[events.digiChamber == ch_num], 'digiEta_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiStrip[events.digiChamber == ch_num], 'digiStrip_ch{}'.format(ch_num))
print("Created event specific lists! ", events.fields)

fig_occu_mask, axs_occu_mask = plt.subplots(4, 4, figsize=(20,20))
fig_occu_mask2, axs_occu_mask2 = plt.subplots(4, 4, figsize=(20,20))
fig_occu, axs_occu = plt.subplots(4, 4, figsize=(20,20))
fig_mult, axs_mult = plt.subplots(4, 4, figsize=(20,20))

for ch_num in ch_list:
    print(ch_num)
    plot_ch = plot_prefix+"ch{}/".format(ch_num)
    if not os.path.exists(plot_ch):
        os.makedirs(plot_ch)
    for eta_num in eta_list:
        print(eta_num)
        plot_eta = plot_ch+"eta{}/".format(eta_num)
        if not os.path.exists(plot_eta):
            os.makedirs(plot_eta)
        #Digi Occupancy
        mask = events['digiEta_ch{}'.format(ch_num)] == eta_num
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        tmp_occu = axs_occu[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        fig_occu.colorbar(tmp_occu[3], label="Number of Entries", ax=axs_occu[eta_num-1, ch_num])
        axs_occu[eta_num-1, ch_num].set_xlabel("BX")
        axs_occu[eta_num-1, ch_num].set_ylabel("Digi Strip")
        axs_occu[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))


        #Digi Occupancy Filtered for hit at BX3 eta2 strip[100,120]
        mask = (events['digiEta_ch{}'.format(ch_num)] == eta_num) & (ak.any((events['digiStrip_ch{}'.format(ch_num)] > 100) & (events['digiStrip_ch{}'.format(ch_num)] < 120) & (events['digiBX_ch{}'.format(ch_num)] == 3) & (events['digiEta_ch{}'.format(ch_num)] == 2), axis=1))
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        tmp_occu_mask = axs_occu_mask[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        fig_occu_mask.colorbar(tmp_occu_mask[3], label="Number of Entries", ax=axs_occu_mask[eta_num-1, ch_num])
        axs_occu_mask[eta_num-1, ch_num].set_xlabel("BX")
        axs_occu_mask[eta_num-1, ch_num].set_ylabel("Digi Strip")
        axs_occu_mask[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))

        """
        for strip_bin_num in range(n_strip_bins):
            strip_start = (strip_bin_num)*(128*3)/n_strip_bins
            strip_end = (strip_bin_num+1)*(128*3)/n_strip_bins
            mask = (events['digiEta_ch{}'.format(ch_num)] == eta_num) & \
                    ((ak.any((events['digiStrip_ch{}'.format(ch_num)] < strip_start) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1) | ak.any((events['digiStrip_ch{}'.format(ch_num)] > strip_end) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1)) == 0)

            x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
            y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

            tmp_occu_multiple = axs_occu_mask2[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
            fig_occu_mask2.colorbar(tmp_occu_mask2[3], label="Number of Entries", ax=axs_occu_mask2[eta_num-1, ch_num])
            axs_occu_mask2[eta_num-1, ch_num].set_xlabel("BX")
            axs_occu_mask2[eta_num-1, ch_num].set_ylabel("Digi Strip")
            axs_occu_mask2[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))
        """


        #Digi Occupancy Filtered for hit ONLY at BX3 eta2 strip[100,120]
        mask = (events['digiEta_ch{}'.format(ch_num)] == eta_num) & \
                (ak.any((events['digiStrip_ch{}'.format(ch_num)] > 100) & (events['digiStrip_ch{}'.format(ch_num)] < 120) & (events['digiBX_ch{}'.format(ch_num)] == 3) & (events['digiEta_ch{}'.format(ch_num)] == 2), axis=1)) & \
                ((ak.any((events['digiStrip_ch{}'.format(ch_num)] < 50) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1) | ak.any((events['digiStrip_ch{}'.format(ch_num)] > 120) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1)) == 0)
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        tmp_occu_mask2 = axs_occu_mask2[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        fig_occu_mask2.colorbar(tmp_occu_mask2[3], label="Number of Entries", ax=axs_occu_mask2[eta_num-1, ch_num])
        axs_occu_mask2[eta_num-1, ch_num].set_xlabel("BX")
        axs_occu_mask2[eta_num-1, ch_num].set_ylabel("Digi Strip")
        axs_occu_mask2[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))



        #Digi Multiplicity
        plot_BX = plot_eta+"BX/"
        if not os.path.exists(plot_BX):
            os.makedirs(plot_BX)
        BX_for_mult = []
        mult_for_mult = []
        for BX in range(16):
            print(BX)
            x = ak.to_numpy(ak.count(events['digiStrip_ch{}'.format(ch_num)][(events['digiEta_ch{}'.format(ch_num)] == eta_num) & (events['digiBX_ch{}'.format(ch_num)] == BX)], axis=1))
            BX_values = np.full_like(x, BX)

            """
            plt.hist(x, bins=200, range=(1,201), log=True)#, bins=[20,400], range=[[30,50],[0,400]])
            plt.xlabel("Digi Multiplicity")
            plt.title("Digi Multiplicity Ch{} Eta{} BX{}".format(ch_num, eta_num, BX))
            plt.savefig(plot_BX+"digi_mult_ch{}_eta{}_BX{}.pdf".format(BX, ch_num, eta_num, BX))
            plt.close()
            """

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


fig_occu.tight_layout()
fig_occu.savefig(plot_prefix+"occupancy.pdf".format(ch_num, eta_num))

fig_occu_mask.tight_layout()
fig_occu_mask.savefig(plot_prefix+"occupancy_masked.pdf".format(ch_num, eta_num))

fig_occu_mask2.tight_layout()
fig_occu_mask2.savefig(plot_prefix+"occupancy_masked2.pdf".format(ch_num, eta_num))

fig_mult.tight_layout()
fig_mult.savefig(plot_prefix+"multiplicity.pdf".format(ch_num, eta_num))
