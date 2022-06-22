#!/bin/bash
poly="/Users/greenberg/Documents/PHD/Projects/Mobility/Parameter_space/big_rivers/big_rivers.gpkg"
mask_method="Jones"     # Jones, Zou,
network_method="grwl"   # grwl, merit, largest
network_path="/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp"
images="false"
masks="true"
mobility="true"
gif="true"
period="quarterly"
out="/Users/greenberg/Documents/PHD/Projects/Mobility/Parameter_space/big_rivers/"

python ../mobility/mobility.py --poly $poly --mask_method $mask_method --network_method $network_method --network_path $network_path --masks $masks --images $images --mobility $mobility --gif $gif --period $period --out $out
