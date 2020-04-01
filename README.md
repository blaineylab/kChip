# kChip Python Package

## Process for analyzing 5-channel data
Find droplets from 5-channel images.
Split the droplets by UV bin
Save each the droplets in each bin separately to a csv file
Import each csv file individually into the Interactive Clustering notebook and assign clusters
Save csv’s from Interactive Clustering for each UV bin
Bring those back into the Analysis notebook, concatenate them, and name the final DataFrame “droplets”
Proceed with post-merge image analysis
Do distance and area filtering
Save the final DataFrame as a csv with the format: “YYYYMMDD_experimentName_trimmed_distance_and_area_filtered.csv”
