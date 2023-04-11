import subprocess

files_to_run = [
    'news_crawl.py',
    'full_content.py',
    'topic_modelling.py',
    'topic_cluster_formation.py',
    'multi_summ.py',
    'graph_summ.py'
]

for file in files_to_run:
    subprocess.run(['python -u', file])
