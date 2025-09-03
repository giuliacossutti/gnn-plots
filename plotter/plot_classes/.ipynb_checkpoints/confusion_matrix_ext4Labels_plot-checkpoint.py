from puma.utils import confusion_matrix
from puma.matshow import MatshowPlot
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import matplotlib.pyplot as plt
from plotter.plot_classes.plotbase import PlotBase

class ConfMatPlotBase(PlotBase):
	"""
	Subclass of PlotBase to plot either jet classification or track origin confusion matrices with 
	extended labelling scheme for track origin.
	"""

	def plot(self):
		# required parameters for vertex index plot base. Set in 'style' key in config
		required_params = {
			'xlabel',
			'ylabel',
			'figsize',
			'fontsize',
		    'label_fontsize',
		    'dpi',
		    'show_entries',
		    'text_color_threshold',
			"use_atlas_tag",
			"atlas_brand",
			"atlas_first_tag",
			"atlas_second_tag",
		    # 'colormap'
		}

		# filter only the necessary parameters from the config file to plot the vertex matrix
		filtered_params = {
		    key: value for key, value in self.config.style.items() if key in required_params
		}

		# extracting sample details and storing as a dictionary
		sample = ConfigDict(self.config.samples)

		# EXTRACTING THE DATA
		# -------------------
		with h5py.File(sample.path, "r") as hdf_file:

			ds_tfj = hdf_file[sample.df_name]

			# get attribute name for GNN ej score
			keys_list = list(ds_tfj.dtype.fields.keys())

            # search for which key contains the GNN signal discriminant
			for i, key in enumerate(keys_list):
				print(key)
				if "GN3ej-fold0_pdisplaced" in key:
					pDisp = keys_list[i]
				elif "GN3ej-fold0_ppileup" in key:
					pPileup = keys_list[i]
				elif "GN3ej-fold0_pfake" in key:
					pFake = keys_list[i]
				elif "GN3ej-fold0_pprimary" in key:
					pPrimary = keys_list[i]
				elif "GN3ej-fold0_pfromBC" in key: #needs to come before pfromB or will never get triggered
					pFromBC = keys_list[i]
				elif "GN3ej-fold0_pfromB" in key:
					pFromB = keys_list[i]
				elif "GN3ej-fold0_pfromC" in key:
					pFromC = keys_list[i]
				elif "GN3ej-fold0_pfromTau" in key:
					pFromTau = keys_list[i]
				elif "GN3ej-fold0_potherSecondary" in key:
					pOtherSecondary = keys_list[i]

			if sample.df_name == 'jets':
				# extract classification labels
				true_class = np.array(ds_tfj['isDisplaced'])
				pred_disp = np.array(ds_tfj[pDisp])
				pred_prompt = 1.0 - pred_disp

				# initialize predicted classification labels
				pred_class = np.empty(len(true_class))

				# update pred_class with most likely predicted jet classification
				for i, (pd, pp) in enumerate(zip(pred_disp, pred_prompt)):
					if pd > pp:
						pred_class[i] = 1  # displaced
					else:
						pred_class[i] = 0  # prompt
				
				# compute the confusion matrix
				confmat = confusion_matrix.confusion_matrix(targets=true_class, predictions=pred_class)

			elif sample.df_name == 'tracks':
				valid = np.array(ds_tfj['valid'])

				# extract valid origin labels
				true_origin = np.array(ds_tfj['truthOriginLabel'])[valid]
				pred_pileup = np.array(ds_tfj[pPileup])[valid]
				pred_fake = np.array(ds_tfj[pFake])[valid]
				pred_primary = np.array(ds_tfj[pPrimary])[valid]
				pred_fromB = np.array(ds_tfj[pFromB])[valid]
				pred_fromBC = np.array(ds_tfj[pFromBC])[valid]
				pred_fromC = np.array(ds_tfj[pFromC])[valid]
				pred_fromTau = np.array(ds_tfj[pFromTau])[valid]
				pred_otherSecondary = np.array(ds_tfj[pOtherSecondary])[valid]
				pred_displaced = np.array(ds_tfj[pDisp])[valid]

				# initialize predicted origin labels
				pred_origin = np.empty(len(true_origin))

				
				
				# update the pred_origin with most likely predicted track origins
				for i, (pu, fk, pr, B, BC, C, Tau, os, dp) in enumerate(zip(pred_pileup, pred_fake, pred_primary, pred_fromB, pred_fromBC, pred_fromC, pred_fromTau, pred_otherSecondary, pred_displaced)):
					origin = max(pu, fk, pr, B, BC, C, Tau, os, dp)
					if origin == pu:
						pred_origin[i] = 0
					elif origin == fk:
						pred_origin[i] = 1
					elif origin == pr:
						pred_origin[i] = 2
					elif origin == dp:      # Old 4 categories
						pred_origin[i] = 3
					'''
					elif origin == B:       # New 9 categories
						pred_origin[i] = 3					
					elif origin == BC:
						pred_origin[i] = 4
					elif origin == C:
						pred_origin[i] = 5
					elif origin == Tau:
						pred_origin[i] = 6
					elif origin == os:
						pred_origin[i] = 7
					elif origin == dp:
						pred_origin[i] = 8
					'''

				print("Max label in predictions:", np.max(pred_origin))
				print("Max label in targets:", np.max(true_origin))
				print("true_origin:", true_origin)

				print("\nNumber of jets:",len(ds_tfj))
				print("\nNumber of tracks in true_origin:", len(true_origin), "\nOf which:\n")
				# Count occurrences of values from 0 to 8
				counts = {i: np.sum(true_origin == i) for i in range(9)}

				# Print the results
				for key, value in counts.items():
					print(f"{key}: {value}")

				print("\nNumber of tracks in pred_origin:", len(pred_origin), "\nOf which:\n")
				# Count occurrences of values from 0 to 8
				pcounts = {i: np.sum(pred_origin == i) for i in range(9)}

				# Print the results
				for key, value in pcounts.items():
					print(f"{key}: {value}")
				
				# compute the confusion matrix
				confmat = confusion_matrix.confusion_matrix(targets=true_origin, predictions=pred_origin)


		# CONSTRUCTING THE FIGURE AND PLOTTING THE CONFUSION MATRIX
		# ---------------------------------------------------------
		confmatplot = MatshowPlot(**filtered_params, x_ticks_rotation=0, colormap=plt.cm.GnBu)

		confmatplot.draw(confmat)
		
		confmatplot.savefig(self.config.file_name, dpi=filtered_params["dpi"])
