# PyTorch-NAQANet
PyTorch implementation of [NAQANet](https://arxiv.org/pdf/1903.00161.pdf) with cross-lingual embeddings

## Answer abilities
The current implementation only handle 'passage_span_extraction' and 'counting answer types  
The logic for 'addition-subtraction' is implemented but note tested.  
'question_span_extraction has not been implemented'  

## Embeddings
We use multilingual fastText Wikipedia word embedding to support multilingual Question Answering.

## Dataset  
The model is trained on [DROP](https://arxiv.org/pdf/1903.00161.pdf) dataset

## Usage  
To train the model on cuda device run  
`python3 train_naqanet.py --use_gpu -g <device_id>`

## Performance
The current implementation with *batch_size* 4, *epochs* 30 reach:  
**F1** = 32.85, **EM** = 29.53  

On both metrics arounf 15 points are lost w.r.t. to the results on paper. This can be explained by the following reasons:
* 2 out of 4 answer abilities have been removed
* Reduced number of layers  size of the encoder stacks before the output layer from 6 to 1.  
  This has been done to avoid out of memory errors
* Original implementation uses GloVE pre-trained embeddings, which proved to perform better

 


