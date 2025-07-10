import pickle
import numpy as np
from typing import Dict, Optional, Sequence
from pathlib import Path

def load_instruction(
    instruction_path: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
):
    if instruction_path is not None:
        with open(instruction_path, "rb") as fid:
            data = pickle.load(fid)
    
