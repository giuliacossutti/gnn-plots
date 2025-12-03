"""Produce histogram of discriminant from ej tagger output and labels"""

import numpy as np
import pandas as pd
import h5py

from ftag import Flavours
from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers, get_good_linestyles
from plotter.config_dict import ConfigDict
from plotter.plot_classes.plotbase import PlotBase

class DiscrimPlotBase(PlotBase):
	"""
	DiscrimPlotBase is a subclass of PlotBase specializing in plotting the histogram of prompt jets'
	and displaced jets' GN2ej scores over all test samples.
	"""

	def plot(self):
		# required parameters for discriminant plot. Set in 'style' key in config
		required_params = {
			'logy',
			'figsize',
			'fontsize',
			'dpi',
			'use_atlas_tag',
			"atlas_brand",
			'atlas_first_tag',
			'atlas_second_tag',
			'label_fontsize'
		}

		# filter only the necessary parameters from the config file to plot the vertex matrix
		filtered_params = {
		    key: value for key, value in self.config.style.items() if key in required_params
		}

		# extracting sample details and storing as a dictionary
		sample = ConfigDict(self.config.samples)

		# extracting data and processing it
		with h5py.File(sample.path, "r") as hdf_file:
			ds = hdf_file['jets']

			keys_list = list(ds.dtype.fields.keys())
			GN2ej_pdispjet = keys_list[-2]

			df = pd.DataFrame({'isDisplaced': np.array(ds['isDisplaced']).transpose(),
							  #'GN2ej_pdispjet': np.array(ds[GN2ej_pdispjet]).transpose()})
							   'GN2ej_pdispjet': np.array(ds["GN3ej-combined-extLabels-fold0-classdict_pdispjet"]).transpose()})
			df = df.dropna()
		    
		    # defining boolean arrays to select the different flavour classes
			is_pu = df["isDisplaced"] == 0
			is_hs = df["isDisplaced"] == 1
		    
			linestyles = get_good_linestyles()[:2]

			# initialize histogram plot
			plot_histo = HistogramPlot(
				bins=np.arange(0,1.001,0.01),
				n_ratio_panels=0,
				xlabel="GN3ej score",
				ylabel="Normalized number of jets",
				leg_ncol=1,
				xmin=self.config.low,
				xmax=self.config.high,
				**filtered_params
			)

			# add the histograms
			plot_histo.add(
				Histogram(
					df[is_pu]["GN2ej_pdispjet"],
					label='Prompt jets',
					colour=Flavours["bjets"].colour,
					linestyle=linestyles[0]
				),
				reference=False
			)
			plot_histo.add(
				Histogram(
					df[is_hs]["GN2ej_pdispjet"],
					label="Emerging jets",
					colour=Flavours["cjets"].colour,
					linestyle=linestyles[1],
				),
				reference=False
			)

			plot_histo.draw()
			plot_histo.savefig(self.config.file_name, transparent=False)