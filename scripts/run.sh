#!/bin/bash
poly="/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/Elwha/Elwha.gpkg"
method="jones"
network_method="merit"
network_path="/Users/greenberg/Documents/PHD/Projects/Mobility/river_networks/channel_networks_full.shp"
images="true"
mobility="true"
gif="true"
out="/Users/greenberg/Documents/PHD/Writing/Mobility_Proposal/GIS/Elwha"

python ../mobility/mobility.py --poly $poly --method $method --network_method $network_method --network_path $network_path --images $images --mobility $mobility --gif $gif --out $out
