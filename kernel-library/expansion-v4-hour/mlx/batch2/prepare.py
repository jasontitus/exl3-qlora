"""Use the reviewed parent renderer for one isolated late source discovery."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
s=importlib.util.spec_from_file_location('late_mlx_prepare',ROOT.parent/'prepare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=ROOT;m.SELECT={};m.main()
