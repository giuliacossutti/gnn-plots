from puma import Histogram, HistogramPlot
from puma.utils import get_good_linestyles
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import pandas as pd
from plotter.plot_classes.plotbase import PlotBase

class SampleInfoPlotBase(PlotBase):
	"""
	Subclass to plot information about samples as a histogram. Specify which information to
	plot in the config file.
	"""

	def plot(self):
		# SET UP HISTOGRAM PLOTBASE
		# -------------------------
		required_params = {
			"ymax",
            "ylabel",
            "xlabel",
            "atlas_second_tag",
            "figsize",
            "logy",
            "y_scale",
        }
		filtered_params = {
        	key: value for key, value in self.config.style.items() if key in required_params
        }
		
		linestyles = get_good_linestyles()[:6]
		
		i = 0
		
		for _, sample in self.config.samples.items():
			sample_config = ConfigDict(sample)
			print(sample_config.path)
			with h5py.File(sample_config.path, "r") as hdf_file:
				if self.config.info_df_name == "jets":
					ds_jet = hdf_file[self.config.info_df_name]
					
					is_disp = ds_jet["isDisplaced"] == 1
					is_prompt = ds_jet["isDisplaced"] == 0
					
					if self.config.style['in_TeV']:
						info = ds_jet[self.config.info_type]/1e6
					else:
						info = ds_jet[self.config.info_type]
						
					info_disp = info[is_disp]
					info_prompt = info[is_prompt]
					
					min_val = min(info)
					max_val = max(info)

					if i == 0:
						info_plot = HistogramPlot(
							bins=np.linspace(min_val,max_val,101), 
							**filtered_params
						)
					
					info_plot.add(
						Histogram(
							info_disp,
							label=f"{sample_config.label}: Emerging Jet",
							linestyle=linestyles[i]
						)
					)
					info_plot.add(
						Histogram(
							info_prompt,
							label=f"{sample_config.label}: QCD Jet",
							linestyle=linestyles[i+1]
						)
					)
					i += 2
				
				elif self.config.info_df_name == "tracks":
					ds_jet = hdf_file["jets"]
					ds_tracks = hdf_file[self.config.info_df_name]

					# determine which jets are EJs or QCD
					is_disp = ds_jet["isDisplaced"] == 1
					# is_disp = is_disp.astype(bool)
					is_prompt = ds_jet["isDisplaced"] == 0
					# is_prompt = np.invert(is_disp)

					# extract track info
					info = ds_tracks[self.config.info_type]

					# parse the data to obtain the EJ track info
					ej = info[is_disp]
					ej_1d = ej.ravel()
					cleaned_ej = ej_1d[~np.isnan(ej_1d)] # get rid of the nan entries
					
					# parse the data to obtain the QCD track info
					qcd = info[is_prompt]
					qcd_1d = qcd.ravel()
					cleaned_qcd = qcd_1d[~np.isnan(qcd_1d)] # get rid of the nan entries

					min_val = min([min(cleaned_ej), min(cleaned_qcd)])
					max_val = max([max(cleaned_ej), max(cleaned_qcd)])
					if np.abs(min_val) < 0.15*max_val:
						min_val = 0
					
					
					# set the plot style
					if i == 0:
						info_plot = HistogramPlot(
							bins=np.linspace(min_val, 10, 11),# ,max_val,self.config.num_bins), 
							**filtered_params
						)
					
					info_plot.add(
						Histogram(
							cleaned_ej,
							label=f"{sample_config.label}: Emerging Jet",
							linestyle=linestyles[i]
						)
					)
					info_plot.add(
						Histogram(
							cleaned_qcd,
							label=f"{sample_config.label}: QCD Jet",
							linestyle=linestyles[i+1]
						)
					)
					i += 2

			info_plot.draw()
			info_plot.savefig(self.config.file_name, transparent=False)

				