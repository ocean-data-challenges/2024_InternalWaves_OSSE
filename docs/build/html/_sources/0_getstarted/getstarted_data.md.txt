# Download the data

<br> 

<br>  

The data are hosted and can be accessed on the MEOM server opendap [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_WOC-ESA/catalog.html). 

## Data information
 

The datasets are available for download with the following structure:

--- 

**``` obs_'REGION'.tar.gz```**

There is a repository for each region: Agulhas, Gulfstream and Mediterranean. In each region repository you'll find an observation zipped file. The ```obs_'Region'/``` repository contains the available observations in this region needed to map currents. The observation data are stored in ```ssh/```, ```sst``` or ```drifters``` sub-repo and consists of netcdf files.  


--- 

**``` DC_product_evaluation.tar.gz```**

The ```DC_product_evaluation/``` repository contains the WOC product performances that have already been evaluated on the data challenges in the form of pyo. The structure of the repository is as follow: 

``` 
.
|-- DC_product_evaluation
|   |-- spectrum/
|   |-- eulerian_rms/
|   |-- cumulative_distance/
|   |-- fronts_vel_comparison  
``` 

You can then check the performance of these products yourself and compare them to your methods following the *DC_example_BFNQG_Agulhas* example. 
 

--- 

**``` drifter.tar.gz```**

The ```drifter/``` repository contains pyo files of the AOML drifter positions in the different regions of interest during the year 2019.  

--- 

**``` lagrangian_position_pickle.tar.gz```**

The ```lagrangian_position_pickle/``` repository contains the AOML corresponding fictive drifter positions using the WOC product currents for comparison.  

---  

**``` DC_example_BFNQG_Agulhas.tar.gz```**

The ```DC_example_BFNQG_Agulhas/``` repository contains all the needed data to run the *DC_example_BFNQG_Agulhas* example.  

--- 


## Download and read the data

The data can be downloaded locally using the wget command. We recommand that the data be stored in the `dc_data/` repository. 
For example, to download and unzip the data needed to run the illustration ```nb_howto_run_metrics/``` notebooks :


```
cd dc_data/
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/MEOM/OCEAN_DATA_CHALLENGES/2024_DC_WOC-ESA/DC_example_BFNQG_Agulhas.tar.gz 
tar -xvf DC_example_BFNQG_Agulhas.tar.gz  
rm -f DC_example_BFNQG_Agulhas.tar.gz
``` 


 

