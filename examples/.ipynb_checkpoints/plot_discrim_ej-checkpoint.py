"""Produce histogram of discriminant from tagger output and labels."""

import numpy as np
from ftag import Flavours

from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers, get_good_linestyles
import h5py
import pandas as pd

file_path = '../salt/salt/logs/GN2ej_20240423-T063340/ckpts/epoch=015-val_loss=0.02852__test_pp_output_val.h5'
with h5py.File(file_path, 'r') as hdf_file:
    ds = hdf_file['jets']

    df = pd.DataFrame({'isDisplaced': np.array(ds['isDisplaced']).transpose(), 'GN2ej_pdispjet': np.array(ds['GN2ej_pdispjet']).transpose()})
    df = df.dropna()
    
    # defining boolean arrays to select the different flavour classes
    is_pu = df["isDisplaced"] == 0
    is_hs = df["isDisplaced"] == 1
    
    linestyles = get_good_linestyles()[:2]

    for val in df[is_pu]["GN2ej_pdispjet"]:
        print(val)
    
    # Initialise histogram plot
    plot_histo = HistogramPlot(
        n_ratio_panels=0,
        ylabel="Normalised number of jets",
        xlabel="GNN discriminant",
        logy=True,
        leg_ncol=1,
        figsize=(5.5, 4.5),
        bins=np.linspace(0, 1.01, 101),
        y_scale=1.5,
        atlas_second_tag="$\\sqrt{s}=13.6$ TeV",
    )
    
    # Add the histograms
    plot_histo.add(
        Histogram(
            df[is_pu]["GN2ej_pdispjet"],
            label="Prompt jets",
            colour=Flavours["bjets"].colour,
            linestyle=linestyles[0],
        ),
        reference=False,
    )
    plot_histo.add(
        Histogram(
            df[is_hs]["GN2ej_pdispjet"],
            label="Emerging jets",
            colour=Flavours["cjets"].colour,
            linestyle=linestyles[1],
        ),
        reference=False,
    )
    plot_histo.draw()
    plot_histo.savefig("histogram_discriminant.png", transparent=False)
