import subprocess

files_to_run = [
    'news_crawl.py',
    'full_content.py',
    'single_summ.py',
    'topic_modelling.py',
    'filter_and_txt_gen.py'
    'graph.py'
]

for file in files_to_run:
    subprocess.run(['python', file])
