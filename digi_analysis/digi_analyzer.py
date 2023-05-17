import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import os

plot_prefix = "plots/run154/"

fname = "../input_files/digi/00000154.root"


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

for ch_num in range(4):
    events = ak.with_field(events, events.OH[events.digiChamber == ch_num], 'OH_ch{}'.format(ch_num))
    events = ak.with_field(events, events.VFAT[events.digiChamber == ch_num], 'VFAT_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiBX[events.digiChamber == ch_num], 'digiBX_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiChamber[events.digiChamber == ch_num], 'digiChamber_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiEta[events.digiChamber == ch_num], 'digiEta_ch{}'.format(ch_num))
    events = ak.with_field(events, events.digiStrip[events.digiChamber == ch_num], 'digiStrip_ch{}'.format(ch_num))
print("Created event specific lists! ", events.fields)

for ch_num in [0,1,2,3]:
    print(ch_num)
    plot_ch = plot_prefix+"ch{}/".format(ch_num)
    if not os.path.exists(plot_ch):
        os.makedirs(plot_ch)
    for eta_num in [1,2,3,4]:
        print(eta_num)
        plot_eta = plot_ch+"eta{}/".format(eta_num)
        if not os.path.exists(plot_eta):
            os.makedirs(plot_eta)
        #Digi Occupancy
        mask = events['digiEta_ch{}'.format(ch_num)] == eta_num
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        plt.hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')#, bins=[20,400], range=[[30,50],[0,400]])
        plt.colorbar(label="Number of Entries")
        plt.xlabel("BX")
        plt.ylabel("Digi Strip")
        plt.title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))
        plt.savefig(plot_eta+"digi_strip_ch{}_eta{}.pdf".format(ch_num, eta_num))
        plt.close()



        #Digi Occupancy Zoomed at BX3 eta2 strip[100,120]
        mask = (events['digiEta_ch{}'.format(ch_num)] == eta_num) & (ak.any((events['digiStrip_ch{}'.format(ch_num)] > 100) & (events['digiStrip_ch{}'.format(ch_num)] < 120) & (events['digiBX_ch{}'.format(ch_num)] == 3), axis=1))
        x = ak.to_numpy(ak.flatten(events['digiBX_ch{}'.format(ch_num)][mask]))
        y = ak.to_numpy(ak.flatten(events['digiStrip_ch{}'.format(ch_num)][mask]))

        plt.hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')
        plt.colorbar(label="Number of Entries")
        plt.xlabel("BX")
        plt.ylabel("Digi Strip")
        plt.title("DigiStrip Ch{} Eta{}".format(ch_num, eta_num))
        plt.savefig(plot_eta+"digi_strip_masked_ch{}_eta{}.pdf".format(ch_num, eta_num))
        plt.close()

        #Digi Multiplicity
        plot_BX = plot_eta+"BX/"
        if not os.path.exists(plot_BX):
            os.makedirs(plot_BX)
        BX_for_mult = []
        mult_for_mult = []
        for BX in range(20):
            print(BX)
            x = ak.to_numpy(ak.count(events['digiStrip_ch{}'.format(ch_num)][(events['digiEta_ch{}'.format(ch_num)] == eta_num) & (events['digiBX_ch{}'.format(ch_num)] == BX)], axis=1))
            BX_values = np.full_like(x, BX)

            plt.hist(x, bins=200, range=(1,201), log=True)#, bins=[20,400], range=[[30,50],[0,400]])
            plt.xlabel("Digi Multiplicity")
            plt.title("Digi Multiplicity Ch{} Eta{} BX{}".format(ch_num, eta_num, BX))
            plt.savefig(plot_BX+"digi_mult_ch{}_eta{}_BX{}.pdf".format(BX, ch_num, eta_num, BX))
            plt.close()

            BX_for_mult.append(BX_values)
            mult_for_mult.append(x)

        BX_for_mult = np.array(BX_for_mult).flatten()
        mult_for_mult = np.array(mult_for_mult).flatten()

        x = BX_for_mult
        y = mult_for_mult

        plt.hist2d(x, y, bins=(20,400), range=((0,20),(0,400)), norm='log')#, bins=[20,400], range=[[30,50],[0,400]])
        plt.colorbar(label="Number of Entries")
        plt.xlabel("BX")
        plt.ylabel("Digi Multiplicity")
        plt.title("Digi Multiplicity Ch{} Eta{}".format(ch_num, eta_num))
        plt.savefig(plot_eta+"digi_mult_ch{}_eta{}.pdf".format(ch_num, eta_num))
        plt.close()




"""
z_distance = 36.0 #mm

for ch_num in range(4):
    z_position = z_distance*ch_num
    events = ak.with_field(events, events.rechitX[events.rechitChamber == ch_num], 'rechitX_ch{}'.format(ch_num))
    events = ak.with_field(events, events.rechitY[events.rechitChamber == ch_num], 'rechitY_ch{}'.format(ch_num))
    events = ak.with_field(events, events.rechitEta[events.rechitChamber == ch_num], 'rechitEta_ch{}'.format(ch_num))
    events = ak.with_field(events, events.rechitEta[events.rechitChamber == ch_num], 'rechitEta_ch{}'.format(ch_num))
    events = ak.with_field(events, ak.full_like(events.rechitX[events.rechitChamber == ch_num], z_position), 'rechitZ_ch{}'.format(ch_num))

#ncount is number of rechits on the chamber
ncount = 1
has_ch0_mask = ak.count(events.rechitX_ch0, axis=1) == ncount
has_ch1_mask = ak.count(events.rechitX_ch1, axis=1) == ncount
has_ch2_mask = ak.count(events.rechitX_ch2, axis=1) == ncount
has_ch3_mask = ak.count(events.rechitX_ch3, axis=1) == ncount

has_ch0ch1ch2_mask = has_ch0_mask & has_ch1_mask & has_ch2_mask
has_ch0ch1ch3_mask = has_ch0_mask & has_ch1_mask & has_ch3_mask
has_ch0ch2ch3_mask = has_ch0_mask & has_ch2_mask & has_ch3_mask
has_ch1ch2ch3_mask = has_ch1_mask & has_ch2_mask & has_ch3_mask

has_ch0ch1ch2ch3_mask = has_ch0_mask & has_ch1_mask & has_ch2_mask & has_ch3_mask

events_for_ch0 = events.mask[has_ch1ch2ch3_mask]
events_for_ch1 = events.mask[has_ch0ch2ch3_mask]
events_for_ch2 = events.mask[has_ch0ch1ch3_mask]
events_for_ch3 = events.mask[has_ch0ch1ch2_mask]


for ch_num, event_list in enumerate([events_for_ch0, events_for_ch1, events_for_ch2, events_for_ch3]):
    chamberZ = ch_num*z_distance
    if ch_num == 0:
        rechitX = ak.concatenate([event_list.rechitX_ch1, event_list.rechitX_ch2, event_list.rechitX_ch3], axis=1)
        rechitZ = ak.concatenate([event_list.rechitZ_ch1, event_list.rechitZ_ch2, event_list.rechitZ_ch3], axis=1)
        prop_eta = event_list.rechitEta_ch1
    if ch_num == 1:
        rechitX = ak.concatenate([event_list.rechitX_ch0, event_list.rechitX_ch2, event_list.rechitX_ch3], axis=1)
        rechitZ = ak.concatenate([event_list.rechitZ_ch0, event_list.rechitZ_ch2, event_list.rechitZ_ch3], axis=1)
        prop_eta = (event_list.rechitEta_ch0 + event_list.rechitEta_ch2)/2.0
    if ch_num == 2:
        rechitX = ak.concatenate([event_list.rechitX_ch0, event_list.rechitX_ch1, event_list.rechitX_ch3], axis=1)
        rechitZ = ak.concatenate([event_list.rechitZ_ch0, event_list.rechitZ_ch1, event_list.rechitZ_ch3], axis=1)
        prop_eta = (event_list.rechitEta_ch1 + event_list.rechitEta_ch3)/2.0
    if ch_num == 3:
        rechitX = ak.concatenate([event_list.rechitX_ch0, event_list.rechitX_ch1, event_list.rechitX_ch2], axis=1)
        rechitZ = ak.concatenate([event_list.rechitZ_ch0, event_list.rechitZ_ch1, event_list.rechitZ_ch2], axis=1)
        prop_eta = event_list.rechitEta_ch2
    chamber_hit = event_list["rechitX_ch{}".format(ch_num)]

    linear_regression = ak.linear_fit(rechitZ, rechitX, axis=1)
    linreg_slope = linear_regression.slope
    linreg_int = linear_regression.intercept

    propagated_X = linreg_slope * chamberZ + linreg_int

    residual_X = chamber_hit - propagated_X

    nTotal = ak.count(residual_X)
    nMatches = ak.sum(residual_X < abs(5))

    events = ak.with_field(events, propagated_X, 'prop_hit_ch{}'.format(ch_num))
    events = ak.with_field(events, linreg_slope, 'prop_slope_ch{}'.format(ch_num))
    events = ak.with_field(events, prop_eta, 'prop_eta_ch{}'.format(ch_num))
    events = ak.with_field(events, residual_X, 'residual_ch{}'.format(ch_num))
    events = ak.with_field(events, ak.any(residual_X < abs(5), axis=1), 'hit_match_ch{}'.format(ch_num))
    events = ak.with_field(events, ak.is_none(event_list) != 1, 'good_quality_prop_ch{}'.format(ch_num))

    plt.hist(events['prop_hit_ch{}'.format(ch_num)][ak.is_none(events['prop_hit_ch{}'.format(ch_num)]) == 0], bins=100, range=(-300,300), log=True)
    plt.xlabel("x [mm]")
    plt.ylabel("Entries")
    plt.title("Propagated Position Chamber {}".format(ch_num))
    plt.savefig(plot_prefix+"prop_position_ch{}.pdf".format(ch_num))
    plt.close()

    plt.hist(events['prop_slope_ch{}'.format(ch_num)][ak.is_none(events['prop_slope_ch{}'.format(ch_num)]) == 0], bins=100, range=(-1,1), log=True)
    plt.xlabel("x/z [mm/mm]")
    plt.ylabel("Entries")
    plt.title("Propagated Slope Chamber {}".format(ch_num))
    plt.savefig(plot_prefix+"prop_slope_ch{}.pdf".format(ch_num))
    plt.close()

    plt.hist(ak.flatten(events['residual_ch{}'.format(ch_num)][ak.is_none(events['residual_ch{}'.format(ch_num)]) == 0]), bins=100, range=(-5,5), log=True)
    plt.xlabel("Residual [mm]")
    plt.ylabel("Entries")
    plt.title("Residual Chamber {}".format(ch_num))
    plt.savefig(plot_prefix+"residual_ch{}.pdf".format(ch_num))
    plt.close()

    plt.hist(events['prop_hit_ch{}'.format(ch_num)][events['hit_match_ch{}'.format(ch_num)]], bins=20, range=(-300,300), log=True, color='green', histtype='step')
    plt.hist(events['prop_hit_ch{}'.format(ch_num)][events['good_quality_prop_ch{}'.format(ch_num)]], bins=20, range=(-300,300), log=True, color='blue', histtype='step')
    plt.xlabel("Propagated Position x [mm]")
    plt.ylabel("Entries")
    plt.title("Efficiency Chamber {}".format(ch_num))
    plt.savefig(plot_prefix+"efficiency_ch{}.pdf".format(ch_num))
    plt.close()


    for eta_num in [1,2,3,4]:
        val_den, bins_den, patches_den = plt.hist(events['prop_hit_ch{}'.format(ch_num)][events['good_quality_prop_ch{}'.format(ch_num)] & ak.any(events['prop_eta_ch{}'.format(ch_num)] == eta_num, axis=1)], bins=20, range=(-300,300), log=True, color='blue', histtype='stepfilled')
        val_num, bins_num, patches_num = plt.hist(events['prop_hit_ch{}'.format(ch_num)][events['hit_match_ch{}'.format(ch_num)] & ak.any(events['prop_eta_ch{}'.format(ch_num)] == eta_num, axis=1)], bins=20, range=(-300,300), log=True, color='green', histtype='step', linewidth=2)
        plt.xlabel("Propagated Position x [mm]")
        plt.ylabel("Entries")
        plt.title("Efficiency Histograms Chamber {} Eta {}".format(ch_num, eta_num))
        plt.savefig(plot_prefix+"efficiency_hists_ch{}_eta{}.pdf".format(ch_num, eta_num))
        plt.close()


        eff = np.divide(val_num, val_den, where=(val_den != 0))
        err = np.divide(val_num * np.sqrt(val_den) + val_den * np.sqrt(val_num), np.power(val_den, 2), where=(val_den != 0))
        plt.errorbar(bins_num[:-1], eff, yerr = err)
        plt.xlabel("Propagated Position x [mm]")
        plt.ylabel("Efficiency")
        plt.title("Efficiency Chamber {} Eta {}".format(ch_num, eta_num))
        plt.savefig(plot_prefix+"efficiency_ch{}_eta{}.pdf".format(ch_num, eta_num))
        #plt.hist(eff, bins=20, range=(-300,300))
        #plt.show()
        plt.close()
"""
