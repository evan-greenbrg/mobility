#!/bin/bash
poly="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Meandering/river_shapes/Castelli1.gpkg"
mask_method="Jones"     # Jones, Zou,
network_method="grwl"   # grwl, merit, largest
network_path="/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp"
images="false"
masks="true"
mobility="true"
gif="true"
period="annual"  # annual, quarterly, bankfull, max, min
out="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Meandering/river_shapes/"
river="Castelli1"

python ../mobility/main.py --poly $poly --mask_method $mask_method --network_method $network_method --network_path $network_path --masks $masks --images $images --mobility $mobility --gif $gif --period $period --out $out --river $river
