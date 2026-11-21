"""casper-fs: the CASPER-t topological feature selector (scikit-learn compatible)."""
from .selector import CasperT
from .discretize import JenksDiscretizer

__all__ = ["CasperT", "JenksDiscretizer"]
__version__ = "0.2.0"
