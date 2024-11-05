# Input specification

<br> 
 

<br> 

Once you have [installed the data challenge](getstarted_install.md) and [downloaded the data](getstarted_data.md), you can now evaluate your maps. Although, you might want to check first that your sea surface currents maps respect a certain format. You can then scroll through the different metrics, checking the "Metrics illustration notebooks" showcasing the BFN-QG in the Agulhas region as an example. 

<br> 

To inform the metric velocity package you must write a json dictionnary for your current maps:




<pre> 
    {
    "data_type": "BFNQG",
    "label": "BFN QG daily",
    "path": "../dc_data/DC_example_BFNQG_Agulhas/maps_BFNQG/",
    "pattern": "BFNQG_WOC_Ugeo_daily_2019",
    "match": "BFNQG_WOC_Ugeo_daily_(\\d{4})-(\\d{2})-(\\d{2}).nc",
    "varu": "ug",
    "varv": "vg",
    "nlon":  "longitude",
    "nlat":  "latitude"
    }
</pre>

<br>   

 