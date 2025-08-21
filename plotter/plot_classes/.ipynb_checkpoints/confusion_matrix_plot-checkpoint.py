from puma.utils import confusion_matrix
from puma.matshow import MatshowPlot
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import matplotlib.pyplot as plt
from plotter.plot_classes.plotbase import PlotBase

class ConfMatPlotBase(PlotBase):
	"""
	Subclass of PlotBase to plot either jet classification or track origin confusion matrices.
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
            'file_name',
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
			task_type = self.config.task_type

			ds_tfj = hdf_file[sample.df_name]

			valid = np.array(ds_tfj['valid'])

			# extract valid origin labels
			true_origin = np.array(ds_tfj['truthOriginLabel'])[valid]
			pred_pileup = np.array(ds_tfj['pileup'])[valid]
			pred_fake = np.array(ds_tfj['fake'])[valid]
			pred_prompt = np.array(ds_tfj['prompt'])[valid]
			pred_disp = np.array(ds_tfj['displaced'])[valid]

			# initialize predicted origin labels
			pred_origin = np.empty(len(true_origin))

			# update the pred_origin with most likely predicted track origins
			for i, (pu, fk, pr, dp) in enumerate(zip(pred_pileup, pred_fake, pred_prompt, pred_disp)):
				origin = max(pu, fk, pr, dp)
				if origin == pu:
					pred_origin[i] = 0
				elif origin == fk:
					pred_origin[i] = 1
				elif origin == pr:
					pred_origin[i] = 2
				elif origin == dp:
					pred_origin[i] = 3

			# compute the confusion matrix
			confmat = confusion_matrix.confusion_matrix(targets=true_origin, predictions=pred_origin)


		# CONSTRUCTING THE FIGURE AND PLOTTING THE CONFUSION MATRIX
		# ---------------------------------------------------------
		confmatplot = MatshowPlot(**filtered_params, x_ticks_rotation=0, colormap=plt.cm.GnBu)

		confmatplot.draw(confmat)

		confmatplot.savefig(self.config.file_name, dpi=filtered_params["dpi"])