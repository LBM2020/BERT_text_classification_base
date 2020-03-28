# BERT_Long_Text_Classification
A version of text classification task using BERT which is implemented by Pytorch.
Beause the max_seq_len of BERT 512,if the length of input sentence is more than 512,it will be cut off,if so,it will be harder to get 
the full semantic.The goal of this version is to improve the effect of long text classification.
The steps are shown as follows:
1)Spliting the input sentence into 'split_num' segments,if the batch_size is 8, the length of sentence is 800,the split_num is 4,the shape
  of split is [8,4,200].In order to meet the needs of BERT,we reshape it to [32,200] namely [8*4,200]
2)Regarding every segment as a full sentence,then we input it to BERT,if we set the max_seq_len of BERT is 256,we will get a shape[32,256]
3)After that,we will use a lstm model to concat the segment,In order to meet the needs of lstm,we reshape the [32,256] into [8,4,256],now,
  we regard it as a sentence which is consists of four words and every word embedding is 256.
  
 PS:if we don't use lstm,we can also add up the segments to get the semantic of the original sentene. 

