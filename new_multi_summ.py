# -*- coding: utf-8 -*-
"""new multi summ.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rr_G1cJmK7yBGCIQFz11XGawo7JdG5Vh
"""

!pip install transformers

!pip install sentencepiece

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer

# Load Pegasus model and tokenizer
model_name = 'google/pegasus-multi_news'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

a="""
The Supreme Court on February 23 allowed former Tamil Nadu Chief Minister Edappadi K. Palaniswami to continue as the interim general secretary of the AIADMK. The apex court has affirmed a Madras High Court Division Bench decision that upheld the conduct of July 2022 general council meeting of the party, during which Mr. Palaniswami was made the party leader and his rival O. Panneerselvam was expelled.
A Bench, led by Justice Dinesh Maheshwari, also directed that an interim order of the apex court on July 6, 2022, in the case was "absolute". The interim order had permitted the July 11 meeting to be held. It had further directed that no restrictions should be placed on the agenda of an earlier General Council meeting held on June 23, 2022.
"""
b="""
Ukrainian President has warned that if China sides with Russia in the war against Ukraine, it would mean World War III. 
President Zelensky thinks the world will hazard a nuclear war to serve his ambition. Perhaps he pins his hope on winning a war with the help of borrowed war machine. Powers, not proxies, fight wars.
Russia does not need military help from any country, as does Ukraine, to fight the war. The Ukraine war is a long-deferred war against the colonial mentality of raising proxies to fight their wars and push their interests. This line of thinking convinced China and India not to sign the US-sponsored condemnation resolution against Russia.
"""
c="""
Congress leader Pawan Khera was arrested at the Delhi airport today, after being taken off a flight to Chhattisgarh capital Raipur, over an alleged insult to Prime Minister Narendra Modi. Nearly 50 Congress leaders launched a rare protest on the tarmac, refusing to let the flight leave. The opposition party also approached the Supreme Court against the arrest.
Pawan Khera, a senior Congress spokesperson, was forced to exit the IndiGo flight after he boarded it as part of a Congress group heading to Raipur for a meeting of the All India Congress Committee (AICC).
"""
documents = [a,b,c]

# Tokenize the documents
batch = tokenizer.prepare_seq2seq_batch(documents, truncation=True, padding='longest', return_tensors='pt')

# Generate the summaries
summary_ids = model.generate(
    batch['input_ids'],
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True
)

# Decode the summaries
summaries = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

# Print the summaries
for i, summary in enumerate(summaries):
    print(f"{summary}")

