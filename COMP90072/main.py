#!/usr/bin/env python3
"""Sandpile Lab
============

This is the main file from which we run all the various routines in the
sandpile lab.

"""
from pathlib import Path

import matplotlib
import numpy as np

from matplotlib import pyplot
from sandpile import SandPile


def main():
    pile = SandPile(50,50, abelian=False)
    pile.drive(drops=50000, site=[25,25],animate_every=0, save_anim=False)
    pile.plot_3D(save_plot=True)
    pile.plot_shape(save_plot=True)
    pile.plot_stats(save_plot=True, save_csv=False)
    

if __name__ == "__main__":
    main()
