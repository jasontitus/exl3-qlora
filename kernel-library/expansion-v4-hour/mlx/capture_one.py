"""One additional measured NAX dispatch case, separate from held main payloads."""
import collect
from pathlib import Path
main=collect.ROOT
original_git=collect.git
original_root=collect.ROOT

def git(repo,*args):
 collect.ROOT=original_root
 try:return original_git(repo,*args)
 finally:collect.ROOT=main/'batch2'
collect.ROOT=main/'batch2';collect.git=git;collect.SELECTION['mlx']=['66a0407']
records=collect.one_repo('mlx');collect.save(collect.ROOT/'inventory-all.json',{'records':records})
