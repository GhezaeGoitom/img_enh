import os
import sys
from pathlib import Path

original_path = str(Path(Path(Path(__file__).parent.absolute()).parent.absolute()).parent.absolute())
sys.path.append(original_path)

from image_enhancement.src.data.MakeDataset import MakeDatasets

# path = MakeDatasets.make_folder(str(os.getcwd))
# print(path)

def test_folder_maker():
    test_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
    path = MakeDatasets.make_folder(test_path)
    assert path != ""


