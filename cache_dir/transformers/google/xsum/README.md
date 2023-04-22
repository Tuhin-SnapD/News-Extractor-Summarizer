---
language: en
tags:
- summarization
model-index:
- name: google/pegasus-xsum
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: train
    metrics:
    - name: ROUGE-1
      type: rouge
      value: 21.8096
      verified: true
    - name: ROUGE-2
      type: rouge
      value: 4.2525
      verified: true
    - name: ROUGE-L
      type: rouge
      value: 17.4469
      verified: true
    - name: ROUGE-LSUM
      type: rouge
      value: 18.8907
      verified: true
    - name: loss
      type: loss
      value: 3.0317161083221436
      verified: true
    - name: gen_len
      type: gen_len
      value: 20.3122
      verified: true
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: xsum
      type: xsum
      config: default
      split: test
    metrics:
    - name: ROUGE-1
      type: rouge
      value: 46.8623
      verified: true
    - name: ROUGE-2
      type: rouge
      value: 24.4533
      verified: true
    - name: ROUGE-L
      type: rouge
      value: 39.0548
      verified: true
    - name: ROUGE-LSUM
      type: rouge
      value: 39.0994
      verified: true
    - name: loss
      type: loss
      value: 1.5717021226882935
      verified: true
    - name: gen_len
      type: gen_len
      value: 22.8821
      verified: true
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: cnn_dailymail
      type: cnn_dailymail
      config: 3.0.0
      split: test
    metrics:
    - name: ROUGE-1
      type: rouge
      value: 22.2062
      verified: true
    - name: ROUGE-2
      type: rouge
      value: 7.6701
      verified: true
    - name: ROUGE-L
      type: rouge
      value: 15.4046
      verified: true
    - name: ROUGE-LSUM
      type: rouge
      value: 19.2182
      verified: true
    - name: loss
      type: loss
      value: 2.681241273880005
      verified: true
    - name: gen_len
      type: gen_len
      value: 25.0234
      verified: true
---

### Pegasus Models
See Docs: [here](https://huggingface.co/transformers/master/model_doc/pegasus.html)

Original TF 1 code [here](https://github.com/google-research/pegasus)

Authors: Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019

Maintained by: [@sshleifer](https://twitter.com/sam_shleifer)

Task: Summarization

The following is copied from the authors' README.

# Mixed & Stochastic Checkpoints

We train a pegasus model with sampled gap sentence ratios on both C4 and HugeNews, and stochastically sample important sentences. The updated the results are reported in this table.

| dataset | C4 | HugeNews | Mixed & Stochastic|
| ---- | ---- | ---- | ----|
| xsum | 45.20/22.06/36.99 | 47.21/24.56/39.25 | 47.60/24.83/39.64|
| cnn_dailymail | 43.90/21.20/40.76 | 44.17/21.47/41.11 | 44.16/21.56/41.30|
| newsroom | 45.07/33.39/41.28 | 45.15/33.51/41.33 | 45.98/34.20/42.18|
| multi_news | 46.74/17.95/24.26 | 47.52/18.72/24.91 | 47.65/18.75/24.95|
| gigaword | 38.75/19.96/36.14 | 39.12/19.86/36.24 | 39.65/20.47/36.76|
| wikihow | 43.07/19.70/34.79 | 41.35/18.51/33.42 | 46.39/22.12/38.41 *|
| reddit_tifu | 26.54/8.94/21.64 | 26.63/9.01/21.60 | 27.99/9.81/22.94|
| big_patent | 53.63/33.16/42.25 | 53.41/32.89/42.07 | 52.29/33.08/41.66 *|
| arxiv | 44.70/17.27/25.80 | 44.67/17.18/25.73 | 44.21/16.95/25.67|
| pubmed | 45.49/19.90/27.69 | 45.09/19.56/27.42 | 45.97/20.15/28.25|
| aeslc | 37.69/21.85/36.84 | 37.40/21.22/36.45 | 37.68/21.25/36.51|
| billsum | 57.20/39.56/45.80 | 57.31/40.19/45.82 | 59.67/41.58/47.59|

The "Mixed & Stochastic" model has the following changes:
- trained on both C4 and HugeNews (dataset mixture is weighted by their number of examples). 
- trained for 1.5M instead of 500k (we observe slower convergence on pretraining perplexity).
- the model uniformly sample a gap sentence ratio between 15% and 45%.
- importance sentences are sampled using a 20% uniform noise to importance scores.
- the sentencepiece tokenizer is updated to be able to encode newline character.


(*) the numbers of wikihow and big_patent datasets are not comparable because of change in tokenization and data:
- wikihow dataset contains newline characters which is useful for paragraph segmentation, the C4 and HugeNews model's sentencepiece tokenizer doesn't encode newline and loose this information.
- we update the BigPatent dataset to preserve casing, some format cleanings are also changed, please refer to change in TFDS.


The "Mixed & Stochastic" model has the following changes (from pegasus-large in the paper):


trained on both C4 and HugeNews (dataset mixture is weighted by their number of examples).
trained for 1.5M instead of 500k (we observe slower convergence on pretraining perplexity).
the model uniformly sample a gap sentence ratio between 15% and 45%.
importance sentences are sampled using a 20% uniform noise to importance scores.
the sentencepiece tokenizer is updated to be able to encode newline character.


Citation
```


@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```