from puma import Histogram, HistogramPlot
from puma.utils import get_good_linestyles
from plotter.config_dict import ConfigDict
import h5py
import numpy as np
import pandas as pd
from plotter.plot_classes.plotbase import PlotBase
from ftag import Cuts

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
			"fontsize",
			"label_fontsize",
			"atlas_tag_outside"
        }
		filtered_params = {
        	key: value for key, value in self.config.style.items() if key in required_params
        }
		
		linestyles = get_good_linestyles()[:6]

		# Minimum and maximum probability of being displaced for selected jets
		pdispjet_min = self.config.pEJ_min
		pdispjet_max = self.config.pEJ_max
		
		i = 0
		
		for _, sample in self.config.samples.items():
			sample_config = ConfigDict(sample)
			print(sample_config.path)
			with h5py.File(sample_config.path, "r") as hdf_file:
				if self.config.info_df_name == "jets":
					ds_jet = hdf_file[self.config.info_df_name]

					# Find which key contains the probability of being displaced
					keys_list = list(ds_jet.dtype.fields.keys())

					for j, key in enumerate(keys_list):
						#print(key)
						if "pdispjet" in key:
						    pDispjet = keys_list[j]

					# Select jets with desired probability of being displaced
					pdispjet_cuts = Cuts.from_list([f"{pDispjet} >= {pdispjet_min}", f"{pDispjet} <= {pdispjet_max}"])
					idx, ds_jet_sel = pdispjet_cuts(ds_jet)

					# Distinguish truth Emerging and Prompt jets
					is_disp = ds_jet["isDisplaced"] == 1
					is_prompt = ds_jet["isDisplaced"] == 0
					is_disp_sel = ds_jet_sel["isDisplaced"] == 1
					is_prompt_sel = ds_jet_sel["isDisplaced"] == 0

					# Choose variable to plot its distribution
					if self.config.style['in_TeV']:
						info = ds_jet[self.config.info_type]/1e6
						info_sel = ds_jet_sel[self.config.info_type]/1e6
					else:
						info = ds_jet[self.config.info_type]
						info_sel = ds_jet_sel[self.config.info_type]
						
					info_disp = info[is_disp]
					info_prompt = info[is_prompt]
					info_disp_sel = info_sel[is_disp_sel]
					info_prompt_sel = info_sel[is_prompt_sel]
					
					min_val = min(info)
					max_val = max(info)

					if i == 0:
						info_plot = HistogramPlot(
							bins=np.linspace(min_val,max_val+1E-5,self.config.num_bins), 
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
					info_plot.add(
						Histogram(
							info_disp_sel,
							label=f"{sample_config.label}: Emerging Jet with ${pdispjet_min} \\leq p_{{\mathrm{{EJ}}}} \\leq {pdispjet_max}$",
							linestyle=linestyles[i+2]
						)
					)
					info_plot.add(
						Histogram(
							info_prompt_sel,
							label=f"{sample_config.label}: QCD Jet with ${pdispjet_min} \\leq p_{{\mathrm{{EJ}}}} \\leq {pdispjet_max}$",
							linestyle=linestyles[i+3]
						)
					)
					i += 4
				
				elif self.config.info_df_name == "tracks":
					ds_jet = hdf_file["jets"][:100000]
					ds_tracks = hdf_file[self.config.info_df_name][:100000]

					# Find which key contains the probability of being displaced
					keys_list = list(ds_jet.dtype.fields.keys())

					for j, key in enumerate(keys_list):
						#print(key)
						if "pdispjet" in key:
						    pDispjet = keys_list[j]

					# Select jets with desired probability of being displaced
					pdispjet_cuts = Cuts.from_list([f"{pDispjet} >= {pdispjet_min}", f"{pDispjet} <= {pdispjet_max}"])
					idx, ds_jet_sel = pdispjet_cuts(ds_jet)
					ds_tracks_sel = ds_tracks[idx]

					# determine which jets are EJs or QCD
					is_disp = ds_jet["isDisplaced"] == 1
					is_prompt = ds_jet["isDisplaced"] == 0
					is_disp_sel = ds_jet_sel["isDisplaced"] == 1
					is_prompt_sel = ds_jet_sel["isDisplaced"] == 0

					# extract track info
					info = ds_tracks[self.config.info_type]
					info_sel = ds_tracks_sel[self.config.info_type]

					# parse the data to obtain the EJ track info
					ej = info[is_disp]
					ej_1d = ej.ravel()
					cleaned_ej = ej_1d[~np.isnan(ej_1d)] # get rid of the nan entries

					ej_sel = info_sel[is_disp_sel]
					ej_1d_sel = ej_sel.ravel()
					cleaned_ej_sel = ej_1d_sel[~np.isnan(ej_1d_sel)] # get rid of the nan entries
					
					# parse the data to obtain the QCD track info
					qcd = info[is_prompt]
					qcd_1d = qcd.ravel()
					cleaned_qcd = qcd_1d[~np.isnan(qcd_1d)] # get rid of the nan entries

					qcd_sel = info_sel[is_prompt_sel]
					qcd_1d_sel = qcd_sel.ravel()
					cleaned_qcd_sel = qcd_1d_sel[~np.isnan(qcd_1d_sel)] # get rid of the nan entries

					min_val = min([min(cleaned_ej), min(cleaned_qcd)])
					max_val = max([max(cleaned_ej), max(cleaned_qcd)])
					if np.abs(min_val) < 0.15*max_val:
						min_val = 0
					
					
					# set the plot style
					if i == 0:
						info_plot = HistogramPlot(
							bins=np.linspace(min_val, max_val+1E-5,self.config.num_bins),# ,max_val,self.config.num_bins), 
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
					info_plot.add(
						Histogram(
							cleaned_ej_sel,
							label=f"{sample_config.label}: Emerging Jet with ${pdispjet_min} \\leq p_{{\mathrm{{EJ}}}} \\leq {pdispjet_max}$",
							linestyle=linestyles[i+2]
						)
					)
					info_plot.add(
						Histogram(
							cleaned_qcd_sel,
							label=f"{sample_config.label}: QCD Jet with ${pdispjet_min} \\leq p_{{\mathrm{{EJ}}}} \\leq {pdispjet_max}$",
							linestyle=linestyles[i+3]
						)
					)
					i += 4

				elif self.config.info_df_name == "num_tracks":
					ds_jet = hdf_file["jets"][:100000]
					ds_tracks = hdf_file["tracks"][:100000]

					# Find which key contains the probability of being displaced
					keys_list = list(ds_jet.dtype.fields.keys())

					for j, key in enumerate(keys_list):
						#print(key)
						if "pdispjet" in key:
						    pDispjet = keys_list[j]

					# Select jets with desired probability of being displaced
					pdispjet_cuts = Cuts.from_list([f"{pDispjet} >= {pdispjet_min}", f"{pDispjet} <= {pdispjet_max}"])
					idx, ds_jet_sel = pdispjet_cuts(ds_jet)
					ds_tracks_sel = ds_tracks[idx]

					# Exclude pileup tracks, if desired
					if self.config.no_pileup:
						nopileup = ds_tracks['truthVertexIndex'] != -2

					# determine which jets are EJs or QCD
					is_disp = ds_jet["isDisplaced"] == 1
					is_prompt = ds_jet["isDisplaced"] == 0
					is_disp_sel = ds_jet_sel["isDisplaced"] == 1
					is_prompt_sel = ds_jet_sel["isDisplaced"] == 0

					# Arrays of number of tracks inside jets
					ej = ds_tracks[is_disp]
					# Exclude pileup tracks, if desired
					if self.config.no_pileup:
						arr_ej = []
						for j in range(0,len(ds_jet[is_disp])):
							a = ej[j]['truthVertexIndex'][ej[j]['valid']]
							a = np.where(a != -2, a, None)
							a = a[a != None]
							arr_ej.append(len(a))
					else:
						arr_ej = [len(ej[i][self.config.info_type][ej[i]['valid']]) for i in range(0,len(ds_jet[is_disp]))]

					ej_sel = ds_tracks_sel[is_disp_sel]
					# Exclude pileup tracks, if desired
					if self.config.no_pileup:
						arr_ej_sel = []
						for j in range(0,len(ds_jet_sel[is_disp_sel])):
							a = ej_sel[j]['truthVertexIndex'][ej_sel[j]['valid']]
							a = np.where(a != -2, a, None)
							a = a[a != None]
							arr_ej_sel.append(len(a))
					else:
						arr_ej_sel = [len(ej_sel[i][self.config.info_type][ej_sel[i]['valid']]) for i in range(0,len(ds_jet_sel[is_disp_sel]))]
					
					qcd = ds_tracks[is_prompt]
					# Exclude pileup tracks, if desired
					if self.config.no_pileup:
						arr_qcd = []
						for j in range(0,len(ds_jet[is_prompt])):
							a = qcd[j]['truthVertexIndex'][qcd[j]['valid']]
							a = np.where(a != -2, a, None)
							a = a[a != None]
							arr_qcd.append(len(a))
					else:
						arr_qcd = [len(qcd[i][self.config.info_type][qcd[i]['valid']]) for i in range(0,len(ds_jet[is_prompt]))]

					qcd_sel = ds_tracks_sel[is_prompt_sel]
					# Exclude pileup tracks, if desired
					if self.config.no_pileup:
						arr_qcd_sel = []
						for j in range(0,len(ds_jet_sel[is_prompt_sel])):
							a = qcd_sel[j]['truthVertexIndex'][qcd_sel[j]['valid']]
							a = np.where(a != -2, a, None)
							a = a[a != None]
							arr_qcd_sel.append(len(a))
					else:
						arr_qcd_sel = [len(qcd_sel[i][self.config.info_type][qcd_sel[i]['valid']]) for i in range(0,len(ds_jet_sel[is_prompt_sel]))]

					min_val = min([min(arr_ej), min(arr_qcd)])
					max_val = max([max(arr_ej), max(arr_qcd)])
					if np.abs(min_val) < 0.15*max_val:
						min_val = 0
					
					
					# set the plot style
					if i == 0:
						info_plot = HistogramPlot(
							bins=np.linspace(min_val, max_val+1E-5,self.config.num_bins),# ,max_val,self.config.num_bins), 
							**filtered_params
						)
					
					info_plot.add(
						Histogram(
							arr_ej,
							label=f"{sample_config.label}: Emerging Jet",
							linestyle=linestyles[i]
						)
					)
					info_plot.add(
						Histogram(
							arr_qcd,
							label=f"{sample_config.label}: QCD Jet",
							linestyle=linestyles[i+1]
						)
					)
					info_plot.add(
						Histogram(
							arr_ej_sel,
							label=f"{sample_config.label}: Emerging Jet with ${pdispjet_min} \\leq p_{{\mathrm{{EJ}}}} \\leq {pdispjet_max}$",
							linestyle=linestyles[i+2]
						)
					)
					info_plot.add(
						Histogram(
							arr_qcd_sel,
							label=f"{sample_config.label}: QCD Jet with ${pdispjet_min} \\leq p_{{\mathrm{{EJ}}}} \\leq {pdispjet_max}$",
							linestyle=linestyles[i+3]
						)
					)
					i += 4

			info_plot.draw()
			info_plot.savefig(self.config.file_name, transparent=False)

				