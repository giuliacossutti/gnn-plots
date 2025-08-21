# from puma import VarVsEff, VarVsEffPlot
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
from atlasify import atlasify
from plotter.plot_classes.plotbase import PlotBase
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class NumVertPerfPlotBase(PlotBase):
	"""
	Subclass of PlotBase to plot signal and background efficiency vs the number of displaced
	or prompt vertices in the event.
	"""

	def plot(self):
		# INITIALIZING FIGURE PLOT BASE
		# -----------------------------

		# required parameters for plot figure
		required_params = {
			"grid",
			"figsize",
			"leg_loc",
			"atlas_second_tag",
			"y_scale",
			"atlas_tag_outside"
		}
		# filtering out necessary parameters from config file
		filtered_params = {
			key: value for key, value in self.config.style.items() if key in required_params
		}

		# define the plots
		plot_sig_eff = VarVsEffPlot(
			mode = "sig_eff",
			ylabel = "Emerging jet efficiency",
			xlabel = r"Number of vertices in jet",
			logy = False,
			**filtered_params
		)

		plot_bkg_rej = VarVsEffPlot(
			mode = "bkg_rej",
			ylabel = "QCD jet rejection",
			xlabel = r"Number of vertices in jet",
			logy = False,
			**filtered_params
		)


		# OPEN OUTPUT DATA FILES AND PROCESS THEM
		# ---------------------------------------
		for _, sample in self.config.samples.items():
			sample_config = ConfigDict(sample)
			with h5py.File(sample_config.path, "r") as hdf_file:
				# jet information dataframe
				ds_jet = pd.DataFrame(hdf_file["jets"][:])
				jet_keys = list(hdf_file["jets"].dtype.fields.keys())

				# track information data (cannot store in dataframe)
				ds_tfj = hdf_file[sample_config.df_name]

				# get the working point
				wp = self.config.working_point

				# string names for probability of displaced and prompt
				pDisp = jet_keys[-2]
				pPrompt = jet_keys[-1]

				# obtain GNN discriminant values
				discs_gnn = ds_jet[pDisp]

				# define boolean arrays to select the different flavour classes
				is_disp = ds_jet["isDisplaced"] == 1
				is_prompt = ds_jet["isDisplaced"] == 0


				# DETERMINE THE NUMBER OF TRUE VERTICES IN EACH SAMPLE
				# ----------------------------------------------------
				num_vertices = []

				# truthVertexIndex ndarray
				truthVI = np.asarray(ds_tfj["truthVertexIndex"])
				
				for i in range(len(truthVI)):
					current = truthVI[i]
					num_vertices.append(len(np.unique(current[current >= 0])))

				num_vertices = np.asarray(num_vertices)


				# DEFINE THE CURVES
				# -----------------
				gnn_ej = VarVsEff(
					x_var_sig = num_vertices[is_disp],
					disc_sig = discs_gnn[is_disp].values,
					x_var_bkg = num_vertices[is_prompt],
					disc_bkg = discs_gnn[is_prompt].values,
					bins = self.config.binedges,
					working_point = None,
					disc_cut = wp,
					label = sample_config.label,
					linewidth = 1.2
				)


				# ADD THE CURVES TO THE PLOTS
				# ---------------------------
				plot_sig_eff.add(gnn_ej)
				plot_sig_eff.leg_loc = self.config.sig_eff_leg_loc
				plot_sig_eff.atlas_second_tag += f", Score > {wp}"

				plot_bkg_rej.add(gnn_ej)
				plot_bkg_rej.leg_loc = self.config.bkg_rej_leg_loc
				plot_bkg_rej.atlas_second_tag += f", Score > {wp}"


		# DRAW AND SAVE THE PLOTS
		plot_sig_eff.draw()
		plot_sig_eff.savefig(self.config.sig_eff_filename, transparent=False)

		plot_bkg_rej.draw()
		plot_bkg_rej.savefig(self.config.bkg_rej_filename, transparent=False)



class NumVertComparePlotBase(PlotBase):
	"""
	Subclass to plot truth number of vertices against model predicted number of vertices
	"""

	def plot(self):
		# INITIALIZING FIGURE PLOT BASE
		# -----------------------------

		# required parameters for plot figure
		required_params = {
			"grid",
			"figsize",
			"dpi",
			"fontsize",
			"leg_fontsize",
			"leg_loc",
			"y_scale",
		}
		# filtering out necessary parameters from config file
		filtered_params = {
			key: value for key, value in self.config.style.items() if key in required_params
		}


		# CONSTRUCTING THE FIGURE
		# -----------------------
		fig = plt.figure(figsize=filtered_params["figsize"], dpi=filtered_params["dpi"])
		gs = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.15, width_ratios=[4,1], height_ratios=[1,4])


		# EXTRACT THE DATA
		# ----------------
		sample_config = ConfigDict(self.config.samples)
		with h5py.File(sample_config.path, "r") as hdf_file:
			# track information data (cannot store in dataframe)
			ds_jet = hdf_file["jets"]
			ds_tfj = hdf_file[sample_config.df_name]

			# DETERMINE THE NUMBER OF TRUE VERTICES IN EACH SAMPLE
			# ----------------------------------------------------
			# truthVertexIndex ndarray
			if self.config.disp_only:
				# obtain signal only jets
				is_disp = np.array(ds_jet["isDisplaced"] == 1)
				truthVI = np.asarray(ds_tfj["truthVertexIndex"])[is_disp]
				truthOrigin = np.asarray(ds_tfj["truthOriginLabel"])[is_disp]
				predVI = np.asarray(ds_tfj["VertexIndex"])[is_disp]
				valid = np.asarray(ds_tfj["valid"])[is_disp]
				pred_pileup = ds_tfj["pileup"][is_disp]
				pred_fake = ds_tfj["fake"][is_disp]
				pred_prompt = ds_tfj["prompt"][is_disp]
				pred_disp = ds_tfj["displaced"][is_disp]
			else:
				# use all jets, both bkg and signal
				truthVI = np.asarray(ds_tfj["truthVertexIndex"])
				truthOrigin = np.asarray(ds_tfj["truthOriginLabel"])
				predVI = np.asarray(ds_tfj["VertexIndex"])
				valid = np.asarray(ds_tfj["valid"])
				pred_pileup = ds_tfj["pileup"]
				pred_fake = ds_tfj["fake"]
				pred_prompt = ds_tfj["prompt"]
				pred_disp = ds_tfj["displaced"]

			# initialize lists to store number of unique vertices
			true_num_vert = []
			pred_num_vert = []

			# loop through samples to determine number of unique vertices
			for i in range(len(truthVI)):
				# calculate number of true unique vertexes in each sample
				true_current = truthVI[i]

				# determine number of unique true dipslaced vertices
				if self.config.disp_only:
					origin_current = truthOrigin[i]
					true_num_vert.append(len(np.unique(true_current[origin_current == 3])))
				# determine number of unique true displaced and prompt vertices
				else:
					true_num_vert.append(len(np.unique(true_current[true_current >= 0])))

				# calculate number of predicted unique vertexes in each sample
				pred_current = predVI[i][valid[i]]
				pu = pred_pileup[i][valid[i]]
				fk = pred_fake[i][valid[i]]
				pr = pred_prompt[i][valid[i]]
				dp = pred_disp[i][valid[i]]

				valid_pred = []	# list of tracks that are predicted to originate from prompt or displaced
				for j in range(len(pred_current)):
					origin = max(pu[j], fk[j], pr[j], dp[j])
					# determine number of unique predicted displaced vertices
					if self.config.disp_only:
						if origin == dp[j]:
							valid_pred.append(pred_current[j])
					# determine number of unique displaced and prompt vertices
					else:
						if origin == pr[j] or origin == dp[j]:
							valid_pred.append(pred_current[j])

				pred_num_vert.append(len(np.unique(valid_pred)))

			true_num_vert = np.asarray(true_num_vert)
			pred_num_vert = np.asarray(pred_num_vert)


			# PLOT THE MAIN DATA IN A SUBFIGURE
			# ----------------------------
			ax_main = fig.add_subplot(gs[1, 0])

			atlasify("Simulation Internal", "$\sqrt{s}=13.6$ TeV, 51.8 fb$^{-1}$", font_size=14, label_font_size=14, sub_font_size=12)

			h, xedges, yedges, im = ax_main.hist2d(
				true_num_vert, 
				pred_num_vert, 
				bins=[np.arange(-0.5,max(true_num_vert)+1.5,1), np.arange(-0.5,max(pred_num_vert)+1.5,1)],
				cmap="Blues",
				density=True
			)
			ax_main.plot(
				[min(true_num_vert), max(true_num_vert)], 
				[min(true_num_vert), max(true_num_vert)],
				"k-"
			)

			# main plot settings
			max_val = self.config.max_val
			xyticks = np.arange(0,max_val+1,5)

			ax_main.set_xlim(0,max_val)
			ax_main.set_ylim(0,max_val)

			ax_main.set_xticks(xyticks)
			ax_main.set_yticks(xyticks)
			ax_main.set_xlabel("True number of vertices", fontsize=filtered_params["fontsize"])
			ax_main.set_ylabel("Predicted number of vertices", fontsize=filtered_params["fontsize"])
			ax_main.minorticks_on()
			ax_main.tick_params(labelsize=filtered_params["fontsize"])

			
			# insert colorbar inside the plot
			cax = inset_axes(ax_main, width="5%", height="70%", loc="center right", borderpad=1)#  bbox_to_anchor=(0.8, 0.1, 0.05, 0.8), bbox_transform=ax_main.transAxes)

			cbar = plt.colorbar(im, cax=cax)

			# cbar.set_label("Number of jets", fontsize=filtered_params["fontsize"], rotation=270, labelpad=10)

			# PLOT THE TRUE VERTEX INDICES HISTOGRAM
			# --------------------------------------
			ax_true = fig.add_subplot(gs[0,0])
			ax_true.hist(true_num_vert, 
				bins=np.arange(-0.5,max(true_num_vert)+1.5,1), 
				density=True,
				color='blue',
				alpha=0.75
				)
			# true vertex histogram plot settings
			ax_true.set_xlim(0,max_val)
			ax_true.set_xticks(xyticks)
			ax_true.minorticks_on()
			ax_true.tick_params(which="both", axis="both", direction="in", left=True, right=True, top=True, bottom=True)
			ax_true.tick_params(labelsize=filtered_params["fontsize"])


			# PLOT THE PREDICTED VERTEX INDICES HISTOGRAM
			# -------------------------------------------
			ax_pred = fig.add_subplot(gs[1,1])
			ax_pred.hist(pred_num_vert, 
				bins=np.arange(-0.5,max(pred_num_vert)+1.5,1), 
				density=True,
				orientation="horizontal", 
				color='blue',
				alpha=0.75)
			# predicted vertex histogram plot settings
			ax_pred.set_ylim(0,max_val)
			ax_pred.set_yticks(xyticks)
			ax_pred.minorticks_on()
			ax_pred.tick_params(which="both", axis="both", direction="in", left=True, right=True, top=True, bottom=True)
			ax_pred.tick_params(labelsize=filtered_params["fontsize"])


			plt.savefig(self.config.filename, 
				dpi=filtered_params["dpi"],
				bbox_inches="tight",
			)

