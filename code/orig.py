"""Imports the reference pipeline that produced the article's numbers
(``../pipeline/gamma_pfi_uhem.py``) as module ``P``.  CASPER_ORIG may point to
another copy of that file."""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("CASPER_ORIG", os.path.join(HERE, "..", "pipeline", "gamma_pfi_uhem.py"))
_spec = importlib.util.spec_from_file_location("P_orig", SRC)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
P.DATA_DIR = os.environ.get("CASPER_DATA", os.path.join(HERE, "data"))
