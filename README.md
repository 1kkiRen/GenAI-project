# Cost-efficient method for LLM adaptation for the Russian language

## About
Our project focuses on adapting a pre-trained large language model (LLM) to a new language with minimal computational costs. Specifically, we explore the adaptation of Llama 3.2 Instruct to Russian using the Token Substitution (TS) approach for modifying tokenizers. Unlike existing works that examine individual adaptation techniques in isolation, our study provides a systematic evaluation by comparing adapted models with LLMs pre-trained on Russian and proprietary models. We assess both text quality and token efficiency using subsets of the SberQuAD dataset and a unified benchmark incorporating oasst2, ru_alpaca, and SynEL samples.

### Done by now
- [x] Collect the dataset
- [x] Train the TS model
- [x] Train modified TS model (partial embeddings freezing)
- [ ] Train modified TS model (partial embeddings freezing with scheduler)
- [ ] Train modified TS model (+ lm_head training)
- [ ] Evaluate the models
- [ ] Write the paper [in progress]

