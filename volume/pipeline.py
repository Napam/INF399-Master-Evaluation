import glob, os 
from cococonvert import convert_csv_labels, convert_csv_outputs
from cocoeval import COCO, COCOeval
import sys


class StdoutRedirection:
    """Standard output redirection context manager"""

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        sys.stdout = open(self._path, mode="w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    mapdir = '/mnt/deepvol/mapdir'

    print("\x1b[33mConverting from CSV to JSON format\x1b[0m")
    convert_csv_labels(os.path.join(mapdir, "nogit_val_labels.csv"), "jsons/nogit_val_labels.json")

    for dir_ in glob.glob(os.path.join(mapdir, "nogit_output_*.csv")):
        print(dir_)
        convert_csv_outputs(dir_, "jsons/"+os.path.basename(dir_).replace('.csv','.json'))
    
    print("\x1b[33mLoading ground truth labels\x1b[0m")
    cocogt = COCO("jsons/nogit_val_labels.json")
    print("\x1b[33mGetting summaries\x1b[0m")
    for dir_ in glob.glob("jsons/nogit_output_*.json"):
        print(f"\x1b[32m{dir_}\x1b[0m")
        cocodt = cocogt.loadRes(dir_)
        cocoEval = COCOeval(cocogt, cocodt, 'bbox3d')
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        with StdoutRedirection("summaries/"+os.path.basename(dir_).replace('.json', '.txt')):
            cocoEval.summarize()
        print()