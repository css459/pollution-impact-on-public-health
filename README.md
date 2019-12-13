# Impacts of Toxic Pollution on Life Expectancy and Cancer Rates for Local Communities
## Cole Smith

# Project Report

A `PDF` rendering is contained in `doc/Project_Report.pdf` and the original **Notebook** is
provided in `notebooks/`.

# Project Poster
 
A `PDF` of the poster is available in `doc/Project Poster/Project_Poster.pdf`

# Layout

All code is structured with PEP-8 documentation in modules under `src/`. The functions
within these modules are also copied into the main notebook under `notebooks/`, but the
structure is retained in `src` for reference.

This code can also run under a Main function instead of a notebook under `src/main.py`.
Procedures are detailed in the main Python file.

## Text files in `doc/`

For technical reference of model performance. Detail in main notebook.

# Requirements

    pip3 install -r requirements.txt
    
For the notebook, there is a tree rendered for which you will need GraphViz binaries
installed to your system.

# Data

A fully-merged data set is available in `data/merged.csv`. However, the RAW data
is not downloaded as it is too big. The sources are provided under `data/raw`.

# Running

Some code is provided in `src/main.py` to view prepared data and see regressions.
However, to run this you **need to run from the root folder** as relative paths
are used.