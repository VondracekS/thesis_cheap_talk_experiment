# Master Thesis - Cheap Talk and Conflicts - Complementary Repository
This repository contains python, R and latex code used for my master's thesis [accessible here](https://is.muni.cz/auth/th/dampf/Thesis_Vondracek_final.pdf))  
My thesis deals with experimental testing of cheap-talk effect in stag-hunt schema game. For more details, please search in the thesis directly.  

## Repository Structure
├───data  
│   ├───attachments  
│   ├───data_original  
│   ├───data_processed  
│   ├───experiment_instructions  
│   └───results_data  
├───plots  
├───text  
└───workflow_scripts  

## Data 
- data_original: data as retrieved from the O-Tree software
- data_processed: outputs of the data wrangling, basically all can be derived from control_full_long.csv and treatment_full_long.csv

## Analysis
- analysis.ipynb: EDA, non-compliant detection, data reformatting
- data_wrangling.ipynb: Necessary data transformations doen in order to obtain a lean data set
- simulation.ipynb: Workflow of game simulation used to obtain simulated data
- modelling.R: Modelling part containing mixed models for simulated data and OLS/GLS models for directly collected data
