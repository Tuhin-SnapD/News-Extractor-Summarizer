"""
This code is a Python script that executes several other Python scripts in sequence using the 
subprocess module. The code defines a list of file names (files_to_run) corresponding to other Python 
scripts that need to be executed. Then, it iterates over each file name in the list and uses 
subprocess.run() to execute them as separate Python processes with the python command.

The scripts that are being executed in sequence are:

The code executes each of these scripts one by one in the order specified in the files_to_run list 
using the subprocess.run() function, which creates separate processes for each script and runs them 
sequentially.
"""

import subprocess

files_to_run = [
    'news_crawl.py',
    'full_content.py',
    'topic_modelling.py',
    'topic_cluster_formation.py',
    'multi_summ.py',
    'graph_summ.py',
    'final.py'
]

for file in files_to_run:
    subprocess.run(['python', file])
