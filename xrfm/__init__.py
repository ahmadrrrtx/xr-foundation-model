"""
XRFM Package — Root init.
"""

__version__ = "1.0.0"
__author__ = "XR Foundation Model Contributors"

from xrfm.config.loader import ConfigLoader
from xrfm.data.loader import XRFMTextDataset, DatasetConfig

__all__ = ["ConfigLoader", "XRFMTextDataset", "DatasetConfig", "__version__"]
