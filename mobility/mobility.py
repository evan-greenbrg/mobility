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


polys = getPolygon(polygon_path, root, name)
