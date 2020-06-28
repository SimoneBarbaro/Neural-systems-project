# Neural systems project on NLP by Francesco Carzaniga, Lorenzo Barbera, Qiuhong Lai, Simone Barbaro, Wenjie He

## Usage

First, load the dataset in the dataset folder and unzip it. Then, you should save the Sent2Vec model to src/sent2vec_model.bin. Finally, ensure all libraries in requirements.txt are installed.

To run the evaluation, execute src/test_information_retrieval.py. The parameters for this file are explained below:
    
--mode: retrieval or pairing. Choose between testing on retrieval of discussion among all relevant papers or only pairing discussions from the same paper.

--k: k for knn prediction of the rankers.

--ranker: ranker model to use, choose among BM25, FastBM25, Sent2Vec, Bert, BM25Hybrid, FastBM25Hybrid, Sent2VecHybrid, BertHybrid

--filter: filtering method, None if not using hybrid ranker, otherwise it can be "and" or "or".

--filter_num_keywords: number of keywords for filtering method.

--tokenizer: tokenizer to use, only some ranker will make use of this parameter, choose among SentenceTokenizer, PuctDigitRemoveTokenizer, split.

--num_grams: max lenght of ngrams for token based rankers, by default it's only 1.

--split: Whether to split document sentences for embedding based rankers.

--rouge: if set, rouge score are reported.

--examples: number of examples to show, both good and bad, by default no examples are shown.
