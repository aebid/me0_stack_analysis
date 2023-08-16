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


class cluster():
    def __init__(self, strips, BXs, chamber, eta, evt_num):
        self.strips = strips
        self.BXs = BXs
        self.chamber = chamber
        self.eta = eta
        self.first_Strip = ak.sort(strips)[0]
        self.first_BX = ak.sort(BXs)[0]
        self.last_BX = ak.sort(BXs)[-1]
        if len(BXs) > 1:
            self.second_BX = ak.sort(BXs)[1]
        else:
            self.second_BX = self.last_BX

        self.cluster_Size = len(strips)
        self.strip = int(np.average(strips))
        self.BX = np.average(BXs)
        self.evt_num = evt_num

#Goal is to cluster the digis within a BX window

ch_list = [0,1,2,3]
eta_list = [1,2,3,4]
BX_window = [1,7]

nEvents = len(events)
print("There are ", nEvents, " events")
ak_cluster_chamber = ak.ArrayBuilder()
ak_cluster_eta = ak.ArrayBuilder()
ak_cluster_strip = ak.ArrayBuilder()
ak_cluster_size = ak.ArrayBuilder()
ak_cluster_BX = ak.ArrayBuilder()
ak_cluster_BX_average = ak.ArrayBuilder()


for evtcount, event in enumerate(events):
    clusters = []
    #if evtcount%1 == 0: print("At event ", evtcount)
    #if evtcount == 100000: break
    if ((evtcount/nEvents)*100)%5 == 0: print("At event ", evtcount)

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

    cChamber = []
    cEta = []
    cStrip = []
    cSize = []
    cBX = []
    cBX_avg = []

    for iCluster in clusters:
        cChamber.append(iCluster.chamber)
        cEta.append(iCluster.eta)
        cStrip.append(iCluster.strip)
        cSize.append(iCluster.cluster_Size)
        cBX.append(iCluster.first_BX)
        cBX_avg.append(iCluster.BX)

    ak_cluster_chamber.append(cChamber)
    ak_cluster_eta.append(cEta)
    ak_cluster_strip.append(cStrip)
    ak_cluster_size.append(cSize)
    ak_cluster_BX.append(cBX)
    ak_cluster_BX_average.append(cBX_avg)


out_file["clusterTree"] = {
        "clusterChamber": ak_cluster_chamber.snapshot(),
        "clusterEta": ak_cluster_eta.snapshot(),
        "clusterStrip": ak_cluster_strip.snapshot(),
        "clusterSize": ak_cluster_size.snapshot(),
        "clusterBX": ak_cluster_BX.snapshot(),
        "clusterBX_avg": ak_cluster_BX_average.snapshot(),
    }




"""
#Testing
a = ak.Array([[1,2,3,5,6,8,9], [4,5], [1], [2], []])
l,r = ak.unzip(ak.combinations(a,2))
al,ar = ak.unzip(ak.argcombinations(a,2))

only_next_cells_mask = (ar == al+1)

l2 = l[only_next_cells_mask]
r2 = r[only_next_cells_mask]

new_cluster_mask = (r2 == l2+1)

shape_list = []
for i in new_cluster_mask:
    value = 1
    for j in i:
        if j == True:
            value += 1
        else:
            shape_list.append(value)
            value = 1
    shape_list.append(value)

#something = [3,2,1,2] #Lengths of each cluster
something = shape_list
clusters = ak.unflatten(a, something, axis=1)
"""
