#!/bin/bash
poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/MeanderingRivers/Shapes/PNG6.gpkg"
mask_method="Jones"     # Jones, Zou,
network_method="grwl"   # grwl, merit, largest
network_path="/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp"    # Needs to be on computer
images="false"
masks="true"
mobility="true"
gif="true"
period="quarterly"  # annual, quarterly, bankfull, max, min
out="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/MeanderingRivers/Shapes/"
river="PNG6"

python ../mobility/main.py --poly $poly --mask_method $mask_method --network_method $network_method --network_path $network_path --masks $masks --images $images --mobility $mobility --gif $gif --period $period --out $out --river $river
