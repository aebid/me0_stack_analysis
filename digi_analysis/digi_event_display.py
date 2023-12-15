import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import norm

#Script to take the unpacked digi file and create plots on the digiBX timing


#Folder to put the plots in
plot_prefix = "plots/run386/"


#Input file name
fname = "../me0_multibx_digi_run386.root"


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



import ROOT


fname = "../me0_multibx_digi_run386.root"

f = ROOT.TFile(fname)
events = f.Get("outputtree")

etalist = [1,2,3,4]

events_to_look_at = [16,32,48]

plot_prefix = "evtdisplay"
if not os.path.exists(plot_prefix):
    os.makedirs(plot_prefix)


#for eventnum in events_to_look_at:
for eventnum in range(16,1000,16):
    c = ROOT.TCanvas("c1", "c1", 1200, 2000)
    c.Divide(1,len(etalist))
    proflist = []
    for eta in etalist:
        c.cd(eta)
        ROOT.gPad.SetGridy()
        profname = "Display Eta {} Event {}".format(eta, eventnum)
        proflist.append(ROOT.TProfile2D(profname, profname, 384, -0.5, 383.5, 6, -0.5, 5.5))
        proflist[eta-1].SetStats(0)
        proflist[eta-1].GetYaxis().SetTitle("ME0 Layer")
        proflist[eta-1].GetXaxis().SetTitle("ME0 Digi Strip")
        events.Project(profname, "digiBX:digiStripChamber:digiStrip", "eventCounter == {} && digiStripEta == {}".format(eventnum, eta))
        proflist[eta-1].Draw("colz TEXT")
    c.SaveAs(plot_prefix+"/evtnum{}.pdf".format(eventnum))
