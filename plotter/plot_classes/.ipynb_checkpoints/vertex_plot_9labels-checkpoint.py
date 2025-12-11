from plotter.config_dict import ConfigDict
import h5py
import numpy as np
from atlasify import atlasify
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from plotter.plot_classes.plotbase import PlotBase
import time
from ftag import Cuts


def make_VImats(true_vi, pred_vi, pred_pileup, pred_fake, pred_primary, pred_fromB, pred_fromBC, pred_fromC, pred_fromTau, pred_otherSecondary, pred_disp):
    """
    Produce both truth and prediction vertex matrices for a specific sample jet. Note that all 
    input arrays assume that non valid tracks have been removed, and arrays have been sorted and
    reorderd based on truth vertex indices.
    
    Parameters:
    ----------
        true_vi: array of truth vertex indices
        pred_vi: array of model predicted vertex indices
        pred_pileup: array of predicted pileup origins
        pred_fake: array of predicted fake origins
        pred_primary: array of predicted primary origins
        pred_fromB: array of predicted from b origins
        pred_fromBC: array of predicted from b->c origins
        pred_fromC: array of predicted from c origins
        pred_fromTau: array of predicted from tau origins
        pred_otherSecondary: array of predicted other secondary origins
        pred_disp: array of predicted displaced origins

    Returns:
    -------
        vi_matrices: list, contains both truth and predicted vertex index plots
    """

    # initialize list to store vertex index matrices
    vi_matrices = []

    n = len(true_vi)

    # initialize vertex index (vi) matrices as completely unpaired
    # note in these matrices: -5 -> not paired, 0+ -> vertex index
    mat_true = np.ones((n,n))*(-5.)
    mat_pred = np.ones((n,n))*(-5.)

    # create truth vi matrices
    for i in range(n):
        # truth vertex: checking if the i^th track is pileup (i.e. not a valid vertex)
        if true_vi[i] == -2:
            mat_true[i][i] = -5.

        # constructing vertex index relationships for valid tracks
        else:
            # check for vi pairs between the i^th and j^th tracks
            for j in range(n):
                # checking for matching vertex pairs
                if true_vi[j] == true_vi[i]:
                    mat_true[i][j] = true_vi[j]

    vi_matrices.append(mat_true)
    
    # create predicted vi matrix
    for i, (pu, fk, pr, B, BC, C, Tau, os, dp) in enumerate(zip(pred_pileup, pred_fake, pred_primary, pred_fromB, pred_fromBC, pred_fromC, pred_fromTau, pred_otherSecondary, pred_disp)):
        # determining predicted origin type of track i
        origin = max(pu, fk, pr, B, BC, C, Tau, os, dp)

        # checking if predicted origin is pilup or fake
        if (origin == pu) or (origin == fk):
            # still have to check if these tracks are predicted to pair with other tracks
            pair = False
            for j in range(n):
                if pred_vi[j] == pred_vi[i]:
                    mat_pred[i][j] = pred_vi[j]
                    if j != i : pair = True
            if pair == True:
                # give it a different value from 0 to distinguish
                # NOTE: THIS IS HERE IF I WANT TO ADD ANOTHER ITEM IN THE LEGEND FOR SHOWING THIS CASE
                mat_pred[i][i] = pred_vi[i]
            else:
                mat_pred[i][i] = -5.

        # checking if predicted origin is prompt
        elif (origin == pr) or (origin == B) or (origin == BC) or (origin == C) or (origin == Tau) or (origin == os) or (origin == dp):
            # check for vi pairs between the i^th and j^th tracks
            for j in range(n):
                if pred_vi[j] == pred_vi[i]:
                    mat_pred[i][j] = pred_vi[j]

    vi_matrices.append(mat_pred)

    # return vi_matrices
    return vi_matrices


class VertexPlotBase(PlotBase):
    def plot(self):
        """
        VertexPlotBase subclass of PlotBase to plot the vertex index matrices for both true
        and predicted.
        """
        print("in plot function")
        # required parameters for vertex index plot base. Set in 'style' key in config
        required_params = {
            'figsize',
            'fontsize',
            'label_fontsize',
            'dpi',
            'show_entries',
            'show_percentages',
            'text_color_threshold',
            "atlas_second_tag",
            "atlas_first_tag",
        }

        # filter only the necessary parameters from the config file to plot the vertex matrix
        filtered_params = {
            key: value for key, value in self.config.style.items() if key in required_params
        }

        # extracting sample details and storing as a dictionary
        sample = ConfigDict(self.config.samples)

        start = time.time()

        print("about to open the h5 file")

        # EXTRACTING THE DATA AND PROCESS IT
        # ----------------------------------
        with h5py.File(sample.path, "r") as hdf_file:
            jet_num = self.config.jet_num
            pdispjet_min = self.config.pEJ_min
            pdispjet_max = self.config.pEJ_max
            is_Disp = self.config.is_Disp

            # extract jet information
            ds_jet = hdf_file['jets'][:100000]
            keys_list = list(ds_jet.dtype.fields.keys())

            # search for which key contains the probability of being displaced
            for i, key in enumerate(keys_list):
                #print(key)
                if "pdispjet" in key:
                   pDispjet = keys_list[i]

            # Select a desired truthness jet with desired probability of being displaced
            pdispjet_cuts = Cuts.from_list([f"{pDispjet} >= {pdispjet_min}", f"{pDispjet} <= {pdispjet_max}"])
            isDisp_cuts = Cuts.from_list([f"isDisplaced == {is_Disp}"])
            combined_cuts = pdispjet_cuts + isDisp_cuts
            idx, ds_jet = combined_cuts(ds_jet)

            truth_isDisp = ds_jet['isDisplaced'][jet_num]
            prob_isDisp = ds_jet[pDispjet][jet_num]
            jet_pt = ds_jet['pt'][jet_num]/1000     # jet transverse momentum in GeV
            jet_eta = ds_jet['eta'][jet_num]

            ds_jet_time = time.time()
            print("finished storing ds_jet data. took time {0:.3f} s".format(ds_jet_time-start))

            print("about to extract ds_tfj info")

            # extract track information
            ds_tfj = hdf_file[sample.df_name][:100000]
            ds_tfj = ds_tfj[idx]

            ds_tfj_jet = ds_tfj[jet_num]  # Load the entire jet_num row once into memory

            ds_tfj_time = time.time()
            print("finished storing ds_tfj data. took time {0:.3f} s".format(ds_tfj_time-ds_jet_time))

            valid = ds_tfj_jet['valid']  # Boolean mask

            # Use NumPy boolean indexing on a single in-memory array
            true_vi_data = ds_tfj_jet['truthVertexIndex'][valid]
            pred_vi_data = ds_tfj_jet['VertexIndex'][valid]

            keys_list_tfj = list(ds_tfj.dtype.fields.keys())

            true_origin_data = ds_tfj_jet['truthOriginLabel'][valid]

            # search for which key contains the GNN signal discriminant
            for i, key in enumerate(keys_list_tfj):
                #print(key)
                if "pdisplaced" in key:
                    pDisp = keys_list_tfj[i]
                elif "ppileup" in key:
                    pPileup = keys_list_tfj[i]
                elif "pfake" in key:
                    pFake = keys_list_tfj[i]
                elif "pprimary" in key:
                    pPrimary = keys_list_tfj[i]
                elif "pfromBC" in key: #needs to come before pfromB or will never get triggered
                    pFromBC = keys_list_tfj[i]
                elif "pfromB" in key:
                    pFromB = keys_list_tfj[i]
                elif "pfromC" in key:
                    pFromC = keys_list_tfj[i]
                elif "pfromTau" in key:
                    pFromTau = keys_list_tfj[i]
                elif "potherSecondary" in key:
                    pOtherSecondary = keys_list_tfj[i]

            pred_pileup_data = ds_tfj_jet[pPileup][valid]
            pred_fake_data = ds_tfj_jet[pFake][valid]
            pred_primary_data = ds_tfj_jet[pPrimary][valid]
            pred_fromB_data = ds_tfj_jet[pFromB][valid]
            pred_fromBC_data = ds_tfj_jet[pFromBC][valid]
            pred_fromC_data = ds_tfj_jet[pFromC][valid]
            pred_fromTau_data = ds_tfj_jet[pFromTau][valid]
            pred_os_data = ds_tfj_jet[pOtherSecondary][valid]
            pred_disp_data = ds_tfj_jet[pDisp][valid]


            ds_trackdata_time = time.time()
            print("finished storing track data. took time {0:.3f} s".format(ds_trackdata_time-ds_tfj_time))

            # sort
            sorted_indices = np.argsort(true_vi_data)

            # sort both true and predicted arrays based on sorted truth vertex index array
            true_vi = true_vi_data[sorted_indices]
            pred_vi = pred_vi_data[sorted_indices]

            # sort track origin data
            true_origin = true_origin_data[sorted_indices]
            pred_pileup = pred_pileup_data[sorted_indices]
            pred_fake = pred_fake_data[sorted_indices]
            pred_primary = pred_primary_data[sorted_indices]
            pred_fromB = pred_fromB_data[sorted_indices]
            pred_fromBC = pred_fromBC_data[sorted_indices]
            pred_fromC = pred_fromC_data[sorted_indices]
            pred_fromTau = pred_fromTau_data[sorted_indices]
            pred_os = pred_os_data[sorted_indices]
            pred_disp = pred_disp_data[sorted_indices]

            # set the view: two options are global and closeup
            if self.config.zoom:
                # adjust matrices based on when no pileup tracks are in truth sample
                for i in range(len(true_vi)):
                    if true_vi[i] != -2:
                        true_vi = true_vi[i:]
                        pred_vi = pred_vi[i:]
                        true_origin = true_origin[i:]
                        pred_pileup = pred_pileup[i:]
                        pred_fake = pred_fake[i:]
                        pred_primary = pred_primary[i:]
                        pred_fromB = pred_fromB[i:]
                        pred_fromBC = pred_fromBC[i:]
                        pred_fromC = pred_fromC[i:]
                        pred_fromTau = pred_fromTau[i:]
                        pred_os = pred_os[i:]
                        pred_disp = pred_disp[i:]
                        break

            n = len(true_vi)

            # create vertex index matrices
            mat_true, mat_pred = make_VImats(
                true_vi, 
                pred_vi, 
                pred_pileup, 
                pred_fake, 
                pred_primary,
                pred_fromB,
                pred_fromBC,
                pred_fromC,
                pred_fromTau,
                pred_os,
                pred_disp
            )


        # CONSTRUCTING THE FIGURE AND PLOTTING THE MATRICES
        # -------------------------------------------------
        fig = plt.figure(figsize=filtered_params['figsize'], dpi=filtered_params['dpi'],)
        gs = gridspec.GridSpec(1, 2, wspace = 0.25)

        # creating the subplots
        ax_true = fig.add_subplot(gs[0,0])

        # add an ATLAS Internal label
        atlasify(filtered_params['atlas_first_tag'], filtered_params['atlas_second_tag'], outside=True, font_size=15, label_font_size=15, sub_font_size=13)

        ax_pred = fig.add_subplot(gs[0,1])
        atlasify(atlas=False, outside=True)


        # plotting the truth and predicted vertex index matrices
        ax_true.imshow(mat_true, cmap=plt.cm.GnBu)
        ax_pred.imshow(mat_pred, cmap=plt.cm.GnBu)


        # ADJUSTING PLOT SETTINGS
        # -----------------------
        # adjust the scatterplot sizes depending on number of valid tracks
        if n <= 30:
            size = 1600/n
        elif (n > 30) and (n < 60):
            size = 1200/n
        elif (n>=60) and (n<120):
            size = 600/n
        else:
            size = 300/n


        ### -----------------------------------------------------------------------------------
        ### DETERMINE THE TRUE ORIGIN TYPES AND PLOT THEM
        ### -----------------------------------------------------------------------------------
        colors = ['gray', 'darkred', 'forestgreen', 'tomato', 'chocolate', 'peachpuff', 'deeppink', 'mediumorchid', 'deepskyblue']

        # plot the truth origin labels
        for i in range(n):
            if true_origin[i] == 0:
                ax_true.scatter(i, i, color=colors[0], marker='o', s=size)

            elif true_origin[i] == 1:
                ax_true.scatter(i, i, color=colors[0], marker='o', s=size)

            elif true_origin[i] == 2:
                ax_true.scatter(i, i, color=colors[2], marker='D', s=np.floor(size*0.6))

            elif true_origin[i] == 3:
                ax_true.scatter(i, i, color=colors[3], marker='^', s=np.floor(size*0.6))

            elif true_origin[i] == 4:
                ax_true.scatter(i, i, color=colors[4], marker='<', s=np.floor(size*0.6))

            elif true_origin[i] == 5:
                ax_true.scatter(i, i, color=colors[5], marker='v', s=np.floor(size*0.6))

            elif true_origin[i] == 6:
                ax_true.scatter(i, i, color=colors[6], marker='P', s=np.floor(size*0.6))

            elif true_origin[i] == 7:
                ax_true.scatter(i, i, color=colors[7], marker='h', s=np.floor(size*0.6))

            elif true_origin[i] == 8:
                ax_true.scatter(i, i, color=colors[8], marker='*', s=np.floor(size*1.1))

            else:
                print("ERROR: write an error message later")



        ### -----------------------------------------------------------------------------------
        ### DETERMINE THE PREDICTED ORIGIN TYPES AND PLOT THEM
        ### -----------------------------------------------------------------------------------
        # add origin information to truth vertex index matrix
        for i, (pu, fk, pr, B, BC, C, Tau, os, dp) in enumerate(zip(pred_pileup, pred_fake, pred_primary, pred_fromB, pred_fromBC, pred_fromC, pred_fromTau, pred_os, pred_disp)):
            origin = max(pu, fk, pr, B, BC, C, Tau, os, dp)
            if origin == pu:
                ax_pred.scatter(i, i, color=colors[0], marker='o', s=size)

            elif origin == fk:
                ax_pred.scatter(i, i, color=colors[0], marker='o', s=size)

            elif origin == pr:
                ax_pred.scatter(i, i, color=colors[2], marker='D', s=np.floor(size*0.6))

            elif origin == B:
                ax_pred.scatter(i, i, color=colors[3], marker='^', s=np.floor(size*0.6))

            elif origin == BC:
                ax_pred.scatter(i, i, color=colors[4], marker='<', s=np.floor(size*0.6))

            elif origin == C:
                ax_pred.scatter(i, i, color=colors[5], marker='v', s=np.floor(size*0.6))

            elif origin == Tau:
                ax_pred.scatter(i, i, color=colors[6], marker='P', s=np.floor(size*0.6))

            elif origin == os:
                ax_pred.scatter(i, i, color=colors[7], marker='h', s=np.floor(size*0.6))

            elif origin == dp:
                ax_pred.scatter(i, i, color=colors[8], marker='*', s=np.floor(size*1.1))

            else:
                print("ERROR: write an error message later")



        ### -----------------------------------------------------------------------------------
        ### CUSTOMIZE AND BUILD THE LEGEND (TO THE RIGHT OF THE PLOT)
        ### ----------------------------------------------------------------------------------
        # Custom legend handles
        handles = [
            plt.Line2D([0], [0], marker='s', markersize=8, color="black", linestyle='None'),
            plt.Line2D([0], [0], marker='o', markersize=8, color=colors[0], linestyle='None'),
            plt.Line2D([0], [0], marker='D', markersize=8, color=colors[2], linestyle='None'),
            plt.Line2D([0], [0], marker='^', markersize=8, color=colors[3], linestyle='None'),
            plt.Line2D([0], [0], marker='<', markersize=8, color=colors[4], linestyle='None'),
            plt.Line2D([0], [0], marker='v', markersize=8, color=colors[5], linestyle='None'),
            plt.Line2D([0], [0], marker='P', markersize=8, color=colors[6], linestyle='None'),
            plt.Line2D([0], [0], marker='h', markersize=8, color=colors[7], linestyle='None'),
            plt.Line2D([0], [0], marker='*', markersize=8, color=colors[8], linestyle='None')
        ]

        labels = ["Vertices", "Pile-up + fake", "Primary", "From B", "From B$\\rightarrow$C", "From C", "From Tau", "Other Secondary", "Displaced"]

        fig.legend(handles=handles, labels=labels, fontsize=15, loc='upper left', 
            bbox_to_anchor=(0.89,0.70), frameon=False, handletextpad=0.05)



        ### -----------------------------------------------------------------------------------
        ### ADJUST THE X/Y LIM AND TICKS
        ### -----------------------------------------------------------------------------------
        # tick positions and labels (x and y share same labels)
        if n <= 51:
            xyticks = np.arange(0,n+1,5)
        elif (n>51) and (n<=101):
            xyticks = np.arange(0,n+1,10)
        else:
            xyticks = np.arange(0,n+1,20)

        # true vertex index matrix plot settings
        fsize = filtered_params['fontsize']
        # ax_true.set_title("Truth Labels")
        ax_true.set_xlim(-0.5, n-0.5)
        ax_true.set_ylim(-0.5, n-0.5)
        ax_true.set_xticks(xyticks)
        ax_true.set_yticks(xyticks)
        ax_true.set_xlabel("Track index ", fontsize=fsize, loc="right")
        ax_true.set_ylabel("Track index ", fontsize=fsize, loc="top")

        ax_true.tick_params(axis="both", which="both", direction="in", labelsize=fsize, right=True, top=True)
        ax_true.tick_params(axis="both", which="major", length=8)
        ax_true.tick_params(axis="both", which="minor", length=5)


        # predicted vertex index matrix plot settings
        ax_pred.set_xlim(-0.5, n-0.5)
        ax_pred.set_ylim(-0.5, n-0.5)
        ax_pred.set_xticks(xyticks)
        ax_pred.set_yticks(xyticks)
        ax_pred.set_xlabel("Track index ", fontsize=fsize, loc="right")
        ax_pred.set_ylabel("Track index ", fontsize=fsize, loc="top")

        ax_pred.tick_params(axis="both", which="both", direction="in", labelsize=fsize, right=True, top=True)
        ax_pred.tick_params(axis="both", which="major", length=6)
        ax_pred.tick_params(axis="both", which="minor", length=3)




        ### -----------------------------------------------------------------------------------
        ### ADD EXTRA TEXT TO THE FIGURE
        ### -----------------------------------------------------------------------------------
        # add text to figure
        if truth_isDisp == 1:
           truthjet = "Truth Emerging Jet"
        elif truth_isDisp == 0:
           truthjet = "Truth Prompt Jet"
         
        text = fig.text(0.92, 0.73,
            f"{truthjet}\nJet $p_{{\mathrm{{T}}}}={jet_pt:.1f}$ GeV\n$p_{{\mathrm{{EJ}}}}=${prob_isDisp:.3f}", ha='left', fontsize=fsize)
        #     r"Jet $p_{{T}}={2:.1f}$ GeV"
        #     "\n"
        #     "\n"
        #     r"$P_{{EJ}}=${3:.3f}".format(jet_num, "Signal" if truth_isDisp == 1 else "QCD", jet_pt, 
        #         prob_isDisp), ha='left', fontsize=fsize-3
        # )

        ax_true.text(0.05, 0.9, "Truth", transform=ax_true.transAxes, fontsize=14, color="black")
        ax_pred.text(0.05, 0.84, "Model\nprediction", transform=ax_pred.transAxes, fontsize=14, color="black")


        plt.savefig(self.config.file_name, dpi=filtered_params['dpi'], bbox_inches='tight')

