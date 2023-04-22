# News Extractor and Summarizer
News Extractor and Summarizer is a Python-based web application that extracts and summarizes news articles. The application uses several Python libraries including NewsAPI, newspaper3k, spacy, requests, [Pegasus](https://huggingface.co/google/pegasus-xsum) from Hugging Face, and a T5 model from Hugging Face's model hub [mrm8488](https://huggingface.co/mrm8488/t5-base-finetuned-news-titles-classification) to classify news articles into different categories (e.g. business, health, science, entertainment). It also includes a graph-based summary feature that uses similarity to summarize multiple documents from topic clusters of CSV. The app is designed to work with news articles in any language supported by NewsAPI.

## Installation
To use this application, you will need to follow the installation steps below:

Clone the repository to your local machine by running 
```
git clone https://github.com/Tuhin-SnapD/News-Extractor-Summarizer.git
cd News-Extractor-Summarizer
pip install -r requirements.txt 
```
Or
```
conda create --name env_name --file requirements.txt
```
## Setup
**To use this project, there are two cache directories with the following structure in the 'News-Extractor-Summarizer' folder:**
```
cache_dir/
├── transformers/
│   ├── google/
│   │   └── xsum/
│   └── mrm8488/
│       └── t5-base-finetuned-news-title-classification/

```

These directories will be used by the Hugging Face Transformers library to cache the pre-trained models and tokenizers.

To complete the setup, please follow these steps:

      - Go to https://huggingface.co/google/pegasus-xsum and download the file 'pytorch_model.bin'.
      - Move the downloaded file 'pytorch_model.bin' into the 'xsum' directory.
      - Next, go to https://huggingface.co/mrm8488/t5-base-finetuned-news-titles-classification and download the file 'pytorch_model.bin'.
      - Move the downloaded file 'pytorch_model.bin' into the 't5-base-finetuned-news-title-classification' directory.

**Rename the file env.template to .env as well as app/config.template.js to app/config.js and replace the placeholder values with your own [NewsAPI Key](https://newsapi.org/) and [Google API Key and Search Engine ID](https://console.developers.google.com/)**


## Usage
Now run the following command

```
python -u main.py 
```

This will run a series of different python scripts available in the directory in order to create various datatsets in a newly made dataset directory.

After datasets have been created run the app/index.html, to view the results in the browser and to google search the final outputs and read more about it.

## Algorithm
![Algo](https://github.com/Tuhin-SnapD/Tuhin-SnapD/blob/main/Algo.png.jpg)

## Contributing
Contributions to this repository are welcome! If you have an idea for a new summarization model or an improvement to an existing one, feel free to create a pull request.

## Acknowledgements
This repository was created by Tuhin and Anant as part of Academic Capstone Project. We would like to thank Prof. Durgesh Kumar and Multiple learned faculties of Vellore Institute of technology.
