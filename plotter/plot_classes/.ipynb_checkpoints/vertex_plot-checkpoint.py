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


def make_VImats(true_vi, pred_vi, pred_pileup, pred_fake, pred_prompt, pred_disp):
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
        pred_prompt: array of predicted prompt origins
        pred_disp: array of predicted displaced origins

    Returns:
    -------
        vi_matrices: list, contains both truth and predicted vertex index plots
    """

    # initialize list to store vertex index matrices
    vi_matrices = []

    n = len(true_vi)

    # initialize vertex index (vi) matrices as completely unpaired
    # note in these matrices: 0 -> track pairs, 1 -> tracks not paired
    mat_true = np.ones((n,n))
    mat_pred = np.ones((n,n))

    # create truth vi matrices
    for i in range(n):
        # truth vertex: checking if the i^th track is pileup (i.e. not a valid vertex)
        if true_vi[i] == -2:
            mat_true[i][i] = 0

        # constructing vertex index relationships for valid tracks
        else:
            # check for vi pairs between the i^th and j^th tracks
            for j in range(n):
                # checking for matching vertex pairs
                if true_vi[j] == true_vi[i]:
                    mat_true[i][j] = 0

    vi_matrices.append(mat_true)
    
    # create predicted vi matrix
    for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
        # determining predicted origin type of track i
        origin = max(pu, fk, pr, dp)

        # checking if predicted origin is pilup or fake
        if (origin == pu) or (origin == fk):
            # still have to check if these tracks are predicted to pair with other tracks
            pair = False
            for j in range(n):
                if pred_vi[j] == pred_vi[i]:
                    mat_pred[i][j] = 0
                    pair = True
            if pair == True:
                # give it a different value from 0 to distinguish
                # NOTE: THIS IS HERE IF I WANT TO ADD ANOTHER ITEM IN THE LEGEND FOR SHOWING THIS CASE
                mat_pred[i][i] = 0.01
            else:
                mat_pred[i][i] = 0

        # checking if predicted origin is prompt
        elif (origin == pr) or (origin == dp):
            # check for vi pairs between the i^th and j^th tracks
            for j in range(n):
                if pred_vi[j] == pred_vi[i]:
                    mat_pred[i][j] = 0

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

            # extract jet information
            ds_jet = hdf_file['jets']
            truth_isDisp = ds_jet['isDisplaced'][jet_num]
            keys_list = list(ds_jet.dtype.fields.keys())
            prob_isDisp = ds_jet[keys_list[-2]][jet_num]
            jet_pt = ds_jet['pt'][jet_num]/1000     # jet transverse momentum in GeV
            jet_eta = ds_jet['eta'][jet_num]

            ds_jet_time = time.time()
            print("finished storing ds_jet data. took time {0:.3f} s".format(ds_jet_time-start))

            print("about to extract ds_tfj info")

            # extract track information
            ds_tfj = hdf_file[sample.df_name]

            ds_tfj_jet = ds_tfj[jet_num]  # Load the entire jet_num row once into memory

            ds_tfj_time = time.time()
            print("finished storing ds_tfj data. took time {0:.3f} s".format(ds_tfj_time-ds_jet_time))

            valid = ds_tfj_jet['valid']  # Boolean mask

            # Use NumPy boolean indexing on a single in-memory array
            true_vi_data = ds_tfj_jet['truthVertexIndex'][valid]
            pred_vi_data = ds_tfj_jet['VertexIndex'][valid]

            true_origin_data = ds_tfj_jet['truthOriginLabel'][valid]
            pred_pileup_data = ds_tfj_jet['pileup'][valid]
            pred_fake_data = ds_tfj_jet['fake'][valid]
            pred_prompt_data = ds_tfj_jet['prompt'][valid]
            pred_disp_data = ds_tfj_jet['displaced'][valid]


            ds_trackdata_time = time.time()
            print("finished storing track data. took time {0:.3f} s".format(ds_trackdata_time-ds_tfj_time))

            '''
            ds_tfj_time = time.time()
            print("finished storing ds_tfj data. took time {0:.3f} s".format(ds_tfj_time-ds_jet_time))

            # boolean array of valid tracks
            valid = ds_tfj['valid'][jet_num]

            # extract vertex index data
            true_vi_data = ds_tfj['truthVertexIndex'][jet_num][valid]
            pred_vi_data = ds_tfj['VertexIndex'][jet_num][valid]

            # extract valid origin labels
            true_origin_data = ds_tfj['truthOriginLabel'][jet_num][valid]
            pred_pileup_data = ds_tfj['pileup'][jet_num][valid]
            pred_fake_data = ds_tfj['fake'][jet_num][valid]
            pred_prompt_data = ds_tfj['prompt'][jet_num][valid]
            pred_disp_data = ds_tfj['displaced'][jet_num][valid]

            ds_trackdata_time = time.time()
            print("finished storing track data. took time {0:.3f} s".format(ds_trackdata_time-ds_tfj_time))
            '''

            # sort
            sorted_indices = np.argsort(true_vi_data)

            # sort both true and predicted arrays based on sorted truth vertex index array
            true_vi = true_vi_data[sorted_indices]
            pred_vi = pred_vi_data[sorted_indices]

            # sort track origin data
            true_origin = true_origin_data[sorted_indices]
            pred_pileup = pred_pileup_data[sorted_indices]
            pred_fake = pred_fake_data[sorted_indices]
            pred_prompt = pred_prompt_data[sorted_indices]
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
                        pred_prompt = pred_prompt[i:]
                        pred_disp = pred_disp[i:]
                        break

            n = len(true_vi)

            # create vertex index matrices
            mat_true, mat_pred = make_VImats(
                true_vi, 
                pred_vi, 
                pred_pileup, 
                pred_fake, 
                pred_prompt,
                pred_disp
            )


        # CONSTRUCTING THE FIGURE AND PLOTTING THE MATRICES
        # -------------------------------------------------
        fig = plt.figure(figsize=filtered_params['figsize'], dpi=filtered_params['dpi'],)
        gs = gridspec.GridSpec(1, 2, wspace = 0.25)

        # creating the subplots
        ax_true = fig.add_subplot(gs[0,0])

        # add an ATLAS Internal label
        atlasify("Simulation", "$\sqrt{s}=13.6$ TeV, 51.8 fb$^{-1}$", outside=True, font_size=15, label_font_size=15, sub_font_size=13)

        ax_pred = fig.add_subplot(gs[0,1])
        atlasify(atlas=False, outside=True)


        # plotting the truth and predicted vertex index matrices
        ax_true.imshow(mat_true, cmap='gray')
        ax_pred.imshow(mat_pred, cmap='gray')


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
        colors = ['gray', 'darkred', 'forestgreen', 'deepskyblue']

        # plot the truth origin labels
        for i in range(n):
            if true_origin[i] == 0:
                ax_true.scatter(i, i, color=colors[0], marker='o', s=size)

            elif true_origin[i] == 1:
                ax_true.scatter(i, i, color=colors[0], marker='o', s=size)

            elif true_origin[i] == 2:
                ax_true.scatter(i, i, color=colors[2], marker='D', s=np.floor(size*0.6))

            elif true_origin[i] == 3:
                ax_true.scatter(i, i, color=colors[3], marker='*', s=np.floor(size*1.1))

            else:
                print("ERROR: write an error message later")



        ### -----------------------------------------------------------------------------------
        ### DETERMINE THE PREDICTED ORIGIN TYPES AND PLOT THEM
        ### -----------------------------------------------------------------------------------
        # add origin information to truth vertex index matrix
        for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
            origin = max(pu, fk, pr, dp)
            if origin == pu:
                ax_pred.scatter(i, i, color=colors[0], marker='o', s=size)

            elif origin == fk:
                ax_pred.scatter(i, i, color=colors[0], marker='o', s=size)

            elif origin == pr:
                ax_pred.scatter(i, i, color=colors[2], marker='D', s=np.floor(size*0.6))

            elif origin == dp:
                ax_pred.scatter(i, i, color=colors[3], marker='*', s=np.floor(size*1.1))

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
            plt.Line2D([0], [0], marker='*', markersize=8, color=colors[3], linestyle='None')
        ]

        labels = ["Vertices", "Pile-up + fake", "Prompt", "Displaced"]

        fig.legend(handles=handles, labels=labels, fontsize=15, loc='center left', 
            bbox_to_anchor=(0.89,0.55), frameon=False, handletextpad=0.05)



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
        text = fig.text(0.92, 0.3,
            f"Jet $p_{{\mathrm{{T}}}}={jet_pt:.1f}$ GeV\n\n$p_{{\mathrm{{EJ}}}}=${prob_isDisp:.3f}", ha='left', fontsize=fsize-3)
        #     r"Jet $p_{{T}}={2:.1f}$ GeV"
        #     "\n"
        #     "\n"
        #     r"$P_{{EJ}}=${3:.3f}".format(jet_num, "Signal" if truth_isDisp == 1 else "QCD", jet_pt, 
        #         prob_isDisp), ha='left', fontsize=fsize-3
        # )

        ax_true.text(0.05, 0.9, "Truth", transform=ax_true.transAxes, fontsize=14)
        ax_pred.text(0.05, 0.84, "Model\nprediction", transform=ax_pred.transAxes, fontsize=14)


        plt.savefig(self.config.file_name, dpi=filtered_params['dpi'], bbox_inches='tight')

