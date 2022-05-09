import argparse
import glob
import os

from matplotlib import pyplot as plt
import numpy as np


method = 'pekel'
polygon_path = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/YellowSmall.gpkg'
root = '/Users/greenberg/Documents/PHD/Projects/Mobility/WaterMaskTesting/monthly'
gif = true
name = 'yellow'

# Submit method -> Make the method call
# Each method call will have:
# 1. getPolygon(s)
# 2. getImages
# 3. getMobility
# 4. makeGif


polys = getPolygon(polygon_path, root, name)
