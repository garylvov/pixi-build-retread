from . import b                  # relative -> MUST be ignored
from .sub import c               # relative -> MUST be ignored
from ..other import d            # relative level 2 -> MUST be ignored
import mypkg.sub                 # first-party absolute -> MUST be ignored
import os, sys, json             # stdlib -> MUST be ignored
import numpy                     # real third-party -> should appear
from scipy import linalg         # real third-party -> should appear
import cv2                       # module != distribution name
try:
    import ujson                 # conditional -> flagged
except ImportError:
    ujson = None
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas                # type-only -> flagged
