# BERT finetuning + CRF
SKTkobert 및 Bert multi-lingual 모델의 finetuning 결과 비교
* reference
   * https://github.com/eagle705/pytorch-bert-crf-ner  

## CRF 
* https://pytorch-crf.readthedocs.io/en/stable/


## data
* train, val set
    * https://github.com/kmounlp/NER 
* test set
    * test_exo_form2.txt :  
    임의의 뉴스에 대하여 Exo-brain 폼으로 주관적으로 작성 (일부 오류 있을 수 있음)
    
## codes
1. [preprocess.ipynb](./preprocess.ipynb) : pre-trained data 생성
    * store bert word embedding at data/bert_data*.pk, data/kobert_data*.pk
2. finetuning
    * BERT_finetune_crf.ipynb
    * KoBERT_finetune_crf_v2_.ipynb
