import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm
import ROOT
from array import array

plot_prefix = "plots/run154/"
#plot_prefix = "plots/run146/"

fname = "../input_files/digi/00000154.root"
#fname = "../input_files/digi/00000146.root"

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


out_file = uproot.recreate("cluster_tree.root")
"""
out_file["clusterTree"] = uproot.TTree(
        {
            "cluster_chamber": "int32",
            "cluster_eta": "int32",
            "cluster_strip": "int32",
            "cluster_size": "int32",
            "cluster_BX": "int32",
            "cluster_BX_average": "float32",
        }
    )
"""

#out_file = ROOT.TFile.Open("cluster_tree.root", "RECREATE")
#out_tree = ROOT.TTree("clusterTree", "clusterTree")
#cChamber = array('f', [0])
#cEta = array('f', [0])
#cStrip = array('f', [0])
#cSize = array('f', [0])
#cBX = array('f', [0])
#cBX_avg = array('f', [0])
#out_tree.Branch('cluster_chamber', cChamber, 'cluster_chamber/F')
#out_tree.Branch('cluster_eta', cEta, 'cluster_eta/F')
#out_tree.Branch('cluster_strip', cStrip, 'cluster_strip/F')
#out_tree.Branch('cluster_size', cSize, 'cluster_size/F')
#out_tree.Branch('cluster_BX', cBX, 'cluster_BX/F')
#out_tree.Branch('cluster_BX_average', cBX_avg, 'cluster_BX_average/F')


class cluster():
    def __init__(self, strips, BXs, chamber, eta, evt_num):
        self.strips = strips
        self.BXs = BXs
        self.chamber = chamber
        self.eta = eta
        self.first_Strip = ak.sort(strips)[0]
        self.first_BX = ak.sort(BXs)[0]
        self.cluster_Size = len(strips)
        self.strip = int(np.average(strips))
        self.BX = np.average(BXs)
        self.evt_num = evt_num

#Goal is to cluster the digis within a BX window

ch_list = [0,1,2,3]
eta_list = [1,2,3,4]
BX_window = [1,7]

#clusters = []

nEvents = len(events)
print("There are ", nEvents, " events")
for evtcount, event in enumerate(events):
    clusters = []
    if ((evtcount/nEvents)*100)%1 == 0: print("At event ", evtcount)
    evt_num = event.eventCounter
    for ch in ch_list:
        for eta in eta_list:
            BX_chamber_eta_mask = (event.digiBX >= BX_window[0]) & (event.digiBX <= BX_window[1]) & (event.digiChamber == ch) & (event.digiEta == eta)
            strips = event.digiStrip[BX_chamber_eta_mask]
            BX = event.digiBX[BX_chamber_eta_mask]

            argsort = ak.argsort(strips)
            strips_sorted = strips[argsort]
            BX_sorted = BX[argsort]

            clusters_strips = []
            clusters_BX = []

            while len(strips_sorted) > 0:
                tmp_cluster = [strips_sorted[0]]
                tmp_BX = [BX_sorted[0]]

                for index, strip in enumerate(strips_sorted):
                    if (((strip - 1) in tmp_cluster) or ((strip + 1) in tmp_cluster)) and (strip not in tmp_cluster):
                        tmp_cluster.append(strip)
                        tmp_BX.append(BX_sorted[index])
                clusters_strips.append(tmp_cluster)
                clusters_BX.append(tmp_BX)

                strips_sorted = strips_sorted[len(tmp_cluster):]
                BX_sorted = BX_sorted[len(tmp_BX):]


            #print(clusters_strips)
            #print(clusters_BX)

            for i in range(len(clusters_strips)):
                clusters.append(cluster(clusters_strips[i], clusters_BX[i], ch, eta, evt_num))

    cChamber = array('f')
    cEta = array('f')
    cStrip = array('f')
    cSize = array('f')
    cBX = array('f')
    cBX_avg = array('f')

    for iCluster in clusters:
        cChamber.append(iCluster.chamber)
        cEta.append(iCluster.eta)
        cStrip.append(iCluster.strip)
        cSize.append(iCluster.cluster_Size)
        cBX.append(iCluster.first_BX)
        cBX_avg.append(iCluster.BX)
    if len(clusters) > 0:
        if "clusterTree" in '\t'.join(out_file.keys()):
            out_file["clusterTree"].extend(
                    {
                        "cluster_chamber": cChamber,
                        "cluster_eta": cEta,
                        "cluster_strip": cStrip,
                        "cluster_size": cSize,
                        "cluster_BX": cBX,
                        "cluster_BX_average": cBX_avg,
                    }
                )
        else:
            out_file["clusterTree"] = {
                        "cluster_chamber": cChamber,
                        "cluster_eta": cEta,
                        "cluster_strip": cStrip,
                        "cluster_size": cSize,
                        "cluster_BX": cBX,
                        "cluster_BX_average": cBX_avg,
                    }


    #if evt_num >= 1000: break

"""

digiBX = events.digiBX[(events.digiBX <= 5) & (events.digiBX >= 2)]
digiChamber = events.digiChamber[(events.digiBX <= 5) & (events.digiBX >= 2)]
digiEta = events.digiEta[(events.digiBX <= 5) & (events.digiBX >= 2)]
digiStrip = events.digiStrip[(events.digiBX <= 5) & (events.digiBX >= 2)]


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

fig_timeres, axs_timeres = plt.subplots(4, figsize=(20,20))

for ch_num in ch_list:
    print(ch_num)
    plot_ch = plot_prefix+"ch{}/".format(ch_num)
    if not os.path.exists(plot_ch):
        os.makedirs(plot_ch)

    x_time_res = np.empty([n_strip_bins*len(eta_list)])
    y_time_res = np.empty([n_strip_bins*len(eta_list)])
    y_time_res_err = np.empty([n_strip_bins*len(eta_list)])
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


        #Digi Occupancy Filtered for hit at BX3 eta3 strip[100,120]
        mask = (events['digiEta_ch{}'.format(ch_num)] == eta_num) & (ak.any((events['digiStrip_ch{}'.format(ch_num)] > 100) & (events['digiStrip_ch{}'.format(ch_num)] < 120) & (events['digiBX_ch{}'.format(ch_num)] == 3) & (events['digiEta_ch{}'.format(ch_num)] == 3), axis=1))
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        tmp_occu_mask = axs_occu_mask[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        fig_occu_mask.colorbar(tmp_occu_mask[3], label="Number of Entries", ax=axs_occu_mask[eta_num-1, ch_num])
        axs_occu_mask[eta_num-1, ch_num].set_xlabel("BX")
        axs_occu_mask[eta_num-1, ch_num].set_ylabel("Digi Strip")
        axs_occu_mask[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))






        #The ideal BX for each eta over strip bins
        #x_time_res = np.empty([n_strip_bins])
        #y_time_res = np.empty([n_strip_bins])
        #y_time_res_err = np.empty([n_strip_bins])
        for strip_bin_num in range(n_strip_bins):
            strip_start = (strip_bin_num)*(128*3)/n_strip_bins
            strip_end = (strip_bin_num+1)*(128*3)/n_strip_bins
            #Look at all digiEvents on eta while in awkward
            mask = events['digiEta_ch{}'.format(ch_num)] == eta_num

            x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
            y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

            #Only look at BX less than 7 and strip within bins in the flat numpy
            mask = (x < 7) & (y >= strip_start) & (y <= strip_end)
            x1 = x[mask]
            y1 = y[mask]

            (mu, sigma) = norm.fit(x1)
            x_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = strip_bin_num + (eta_num-1)*n_strip_bins
            y_time_res[strip_bin_num + (eta_num-1)*n_strip_bins] = mu
            y_time_res_err[strip_bin_num + (eta_num-1)*n_strip_bins] = sigma
            #print("Mu/Sig = ", mu, sigma)
            #The BX is only valid from 2 to 6 in this case
            #print("Making the test for bins ", strip_start, strip_end, " and eta/ch ", eta_num, ch_num)
            #plt.hist(x1, bins=4, range=(2,6))
            #plt.savefig("testing.pdf")
            #plt.close()
            #exit()



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

            #plt.hist(x, bins=200, range=(1,201), log=True)#, bins=[20,400], range=[[30,50],[0,400]])
            #plt.xlabel("Digi Multiplicity")
            #plt.title("Digi Multiplicity Ch{} Eta{} BX{}".format(ch_num, eta_num, BX))
            #plt.savefig(plot_BX+"digi_mult_ch{}_eta{}_BX{}.pdf".format(BX, ch_num, eta_num, BX))
            #plt.close()

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

    tmp_time_res = axs_timeres[ch_num].errorbar(x_time_res, y_time_res, yerr=y_time_res_err)
    axs_timeres[ch_num].set_xlabel("Strip Bin (x4)")
    axs_timeres[ch_num].set_ylabel("Mean BX")
    axs_timeres[ch_num].set_title("Time Resolution Ch{}".format(ch_num, eta_num))
    axs_timeres[ch_num].set_ylim(1,5)


fig_occu.tight_layout()
fig_occu.savefig(plot_prefix+"occupancy.pdf".format(ch_num, eta_num))

fig_occu_mask.tight_layout()
fig_occu_mask.savefig(plot_prefix+"occupancy_masked.pdf".format(ch_num, eta_num))

fig_occu_mask2.tight_layout()
fig_occu_mask2.savefig(plot_prefix+"occupancy_masked2.pdf".format(ch_num, eta_num))

fig_mult.tight_layout()
fig_mult.savefig(plot_prefix+"multiplicity.pdf".format(ch_num, eta_num))

fig_timeres.tight_layout()
fig_timeres.savefig(plot_prefix+"time_res.pdf")


strip_bins_mask = 3


for strip_bin_mask_num in range(strip_bins_mask):
    strip_low = (strip_bin_mask_num)*(128*3/strip_bins_mask)
    strip_high = (strip_bin_mask_num+1)*(128*3/strip_bins_mask)
    fig_strip_masks, axs_strip_masks = plt.subplots(4, 4, figsize=(20,20))

    for ch_num in ch_list:
        for eta_num in eta_list:

            eta_mask_num = 3


            #Digi Occupancy Filtered for hit at BX3 eta{eta} strip{strip_low, strip_high}
            mask = (events['digiEta_ch{}'.format(ch_num)] == eta_num) & \
                    ((ak.any((events['digiStrip_ch{}'.format(ch_num)] < strip_low) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1) | ak.any((events['digiStrip_ch{}'.format(ch_num)] > strip_high) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1)) == 0)
            x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
            y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

            tmp_strip_masks = axs_strip_masks[eta_num-1, ch_num].hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
            fig_strip_masks.colorbar(tmp_strip_masks[3], label="Number of Entries", ax=axs_occu_mask2[eta_num-1, ch_num])
            axs_strip_masks[eta_num-1, ch_num].set_xlabel("BX")
            axs_strip_masks[eta_num-1, ch_num].set_ylabel("Digi Strip")
            axs_strip_masks[eta_num-1, ch_num].set_title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))


    fig_strip_masks.tight_layout()
    fig_strip_masks.savefig(plot_prefix+"strip_mask_{}_{}.pdf".format(strip_low, strip_high))


"""
