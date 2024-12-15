import streamlit as st
import os
import pytesseract
from pdf2image import convert_from_path
import tempfile
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from PIL import Image
import math
from typing import Tuple, List, Dict, Optional

[... весь код, который вы предоставили ...]