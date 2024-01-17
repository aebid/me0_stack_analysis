import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm
import ROOT
from array import array

plot_prefix = "plots/run416/"

fname = "../me0_multibx_digi_run386.root"
fname = "../me0_multibx_digi_run400.root"
fname = "../me0_multibx_digi_run401.root"
#fname = "../me0_multibx_digi_run403.root"
fname = "../me0_multibx_digi_run416.root"

outname = "../awk_cluster_run386.root"
outname = "../awk_cluster_run400.root"
outname = "../awk_cluster_run401.root"
#outname = "../awk_cluster_run403.root"
outname = "../awk_cluster_run416_test.root"


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


out_file = uproot.recreate(outname)

"""
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
"""

#Goal is to cluster the digis within a BX window

ch_list = [0,1,2,3]
eta_list = [1,2,3,4]
BX_window = [1,7]

fill_none_value = 9999

nEvents = len(events)
print("There are ", nEvents, " events")


#To do everything all at once, we need to prepare the 'clusters' in order. To do this I make a new array 'sort values'
#sort_values just creates an index, ch:eta:strip as
#sort_values = ch*10000 + eta*1000 + strips
#This will then separate everything by chamber as a number sort, then by eta, then strips
#Ch 1, Eta 3, Strips [1,2,3] vs Ch 3, Eta 2, Strips [3,4,5,6] vs Ch 3, Eta 1, Strips [2,3,4,5]
#sort_values = [13001, 13002, 13003, 32003, 32004, 32005, 32006, 31002, 31003, 31004, 31005]
#We do not care about the BX order!!!

BX_mask = (events.digiBX > BX_window[0]) & (events.digiBX < BX_window[1])

digiChamber = events.digiStripChamber[BX_mask]
digiEta = events.digiStripEta[BX_mask]
digiBX = events.digiBX[BX_mask]
digiStrip = events.digiStrip[BX_mask]

sort_values = ak.argsort(digiChamber*10000 + digiEta*1000 + digiStrip)
digiChamber = digiChamber[sort_values]
digiEta = digiEta[sort_values]
digiBX = digiBX[sort_values]
digiStrip = digiStrip[sort_values]


strips = ak.pad_none(digiStrip, 1)
BXs = ak.pad_none(digiBX, 1)
chambers = ak.pad_none(digiChamber, 1)
etas = ak.pad_none(digiEta, 1)

l,r = ak.unzip(ak.combinations(strips,2))
al,ar = ak.unzip(ak.argcombinations(strips,2))

only_next_cells_mask = (ar == al+1)

#only_next_cells_AND_same_cheta_mask = (ar == al+1) & (digiChamber[ar] == digiChamber[al]) & (digiEta[ar] == digiEta[al])

l2 = l[only_next_cells_mask]
r2 = r[only_next_cells_mask]
al2 = al[only_next_cells_mask]
ar2 = ar[only_next_cells_mask]

new_cluster_mask = (r2 == l2+1) & (digiChamber[ar2] == digiChamber[al2]) & (digiEta[ar2] == digiEta[al2])

shape_list = []

#Finds the cuts of the cluster objects, but I couldn't do it in a smart way so I had to use a for loop
for count, i in enumerate(new_cluster_mask):
    if count%10000 == 1: print("At count ", count)
    value = 1
    for j in i:
        if j == True:
            value += 1
        else:
            shape_list.append(value)
            value = 1
    shape_list.append(value)


clusterStrips = ak.unflatten(strips, shape_list, axis=1)
clusterBXs = ak.unflatten(BXs, shape_list, axis=1)
clusterChambers = ak.unflatten(chambers, shape_list, axis=1)
clusterEtas = ak.unflatten(etas, shape_list, axis=1)

#Remove nones
clusterStrips = ak.fill_none(clusterStrips[~ak.is_none(clusterStrips, axis=2)], fill_none_value)
clusterBXs = ak.fill_none(clusterBXs[~ak.is_none(clusterBXs, axis=2)], fill_none_value)
clusterChambers = ak.fill_none(clusterChambers[~ak.is_none(clusterChambers, axis=2)], fill_none_value)
clusterEtas = ak.fill_none(clusterEtas[~ak.is_none(clusterEtas, axis=2)], fill_none_value)



clusterStrip_min = ak.fill_none(ak.min(clusterStrips, axis=2), fill_none_value)
clusterStrip_avg = ak.fill_none(ak.mean(clusterStrips, axis=2), fill_none_value)
clusterStrip_size = ak.fill_none(ak.num(clusterStrips, axis=2), fill_none_value)

clusterBX_min = ak.fill_none(ak.min(clusterBXs, axis=2), fill_none_value)
clusterBX_max = ak.fill_none(ak.max(clusterBXs, axis=2), fill_none_value)

tmp_seconds = ak.pad_none(ak.sort(clusterBXs, axis=2), 2, axis=2)
clusterBX_second = ak.where(
        ak.is_none(tmp_seconds[:,:,1], axis=1),
            tmp_seconds[:,:,0],
            tmp_seconds[:,:,1]
    )
clusterBX_second = ak.fill_none(clusterBX_second, fill_none_value)
#clusterBX_second = ak.pad_none(ak.sort(clusterBXs, axis=2), 2, axis=2)[:,:,1]
clusterBX_avg = ak.fill_none(ak.mean(clusterBXs, axis=2), fill_none_value)

clusterBX_center = ak.flatten(clusterBXs[clusterStrips == ak.fill_none(ak.values_astype(ak.mean(clusterStrips, axis=2), "int64"), fill_none_value)], axis=2)


clusterChamber = ak.fill_none(ak.mean(clusterChambers, axis=2), fill_none_value)
clusterEta = ak.fill_none(ak.mean(clusterEtas, axis=2), fill_none_value)


out_file["clusterTree"] = {
        "clusterChamber": clusterChamber,
        "clusterEta": clusterEta,
        "clusterStrip": clusterStrip_avg,
        "clusterSize": clusterStrip_size,
        "clusterBX_first": clusterBX_min,
        "clusterBX_second": clusterBX_second,
        "clusterBX_last": clusterBX_max,
        "clusterBX_avg": clusterBX_avg,
        "clusterBX_center": clusterBX_center,
    }
