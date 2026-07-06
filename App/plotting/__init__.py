"""Plotting / matplotlib presentation layer (depends on shared App models via ``..``)."""

from .figure_host import SignalFigureHost
from .plot_overlay import Area, Shades
from .presenter import PlotPresenter
from .spectrogram_settings import SpectrogramSettings

__all__ = ["Area", "PlotPresenter", "Shades", "SignalFigureHost", "SpectrogramSettings"]
