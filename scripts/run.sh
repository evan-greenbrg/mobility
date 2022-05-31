#!/bin/bash
poly="/Users/greenberg/Documents/PHD/Projects/Mobility/Parameter_space/52522/Purus.gpkg"
mask_method="Jones"     # Jones, Zou,
network_method="merit"   # grwl, merit, largest
network_path="/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp"
images="true"
mobility="true"
gif="true"
out="/Users/greenberg/Documents/PHD/Projects/Mobility/Parameter_space/52522"

python ../mobility/mobility.py --poly $poly --mask_method $mask_method --network_method $network_method --network_path $network_path --images $images --mobility $mobility --gif $gif --out $out
