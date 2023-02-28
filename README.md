# The code and data related to the paper titled "Complexity of spatiotemporal plantar pressure patterns during everyday behaviours"
Preprint available on BioRxiv at https://doi.org/10.1101/2023.01.27.525870

## Paper authors:
* Luke D Cleland<sup>1,3*</sup> - ldcleland1@sheffield.ac.uk - repository manager
* Holly M Rowland<sup>1</sup> - hmrowland1@sheffield.ac.uk
* Claudia Mazzà<sup>2,3</sup> - c.mazza@sheffield.ac.uk
* Hannes P Saal<sup>1,3</sup> - h.saal@sheffield.ac.uk

<sup>1</sup> Department of Psychology, The University of Sheffield, UK <br />
<sup>2</sup> Department of Mechanical Engineering, The University of Sheffield, UK <br />
<sup>3</sup> Insigneo institute for in silico medicine, The University of Sheffield, UK <br />
<sup>*</sup> Corresponding author

### Repository author
All code within this repository is authored by Luke Cleland unless otherwise specified.
Within `insole.py`, `import_data()`, `cut_frame()`, `map2footsim()` are all adapted from FootSim. 
`code/project_insole.m` is adapted from code used in other studies conducted by colleagues within the Department of Mechanical Engineering, University of Sheffield

### Contents of the repository
* `/administration` contains the project checklist with task descriptions, consent form and information sheet
* `/code` contains all files of code required to collate, process and analyse the data
* `/preprocessed_data` contains preprocessed data that is used for data analysis 
* `/individual_figures` contains the individual figures created from `figures.py`
* `/paper_figures` contains the panels found within the manuscript
* `/scaled_data` folder to store participant insole data scaled to a common foot size
* `/processed_data` is the location that files with pre-processed data in will be saved following the processing pipeline. These files are used for later analysis and figure generations

### Data on the Open Science Framework (https://osf.io/n9f8w/)
* `/raw_data` contains all raw data files relating to IMU recordings and insole recordings. Should be saved in a folder named `raw_data` within this repository
     - `/participant id`
          - `/trial id`
               - `/.csv` raw data files
               - `/.fsx` TekScan insole recordings
               - `/.h5` IMU sensor recordings
* `/processed_data` contains files with pre-processed data in will be saved following the processing pipeline. 

## Versions:
* Python 3.8.2
* Numpy 1.19.1
* Pandas 1.1.3
* Matplotlib 3.3.1
* Seaborn 0.11.1
* Scipy 1.5.0
* Scikit-learn - 1.1.1    
* Scikit-image - 0.16.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
* FootSim - date 20/10/2022


## Additional files required
FootSim is required to process the raw data. Public FootSim repository can be found at https://github.com/ActiveTouchLab/footsim-python.
Code for "Complexity of spatiotemporal plantar pressure patterns during everyday behaviours" is built upon that in the FootSim repository

## Acknowledgements
LC is supported by a studentship from the MRC Discovery Medicine North (DiMeN)
Doctoral Training Partnership (MR/N013840/1).
