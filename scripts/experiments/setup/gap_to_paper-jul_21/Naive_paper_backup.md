## Naive paper:
We formulate the problem as follows: Given D pretraining datasets and E downstream evaluation datasets with T tasks, [what is the best way to pretrain in order to get the highest performance.](`This needs to be fixed Best cannot be proven`)

We train an open 50M parameter model on an 8-way graph curriculum, and test it on 4 node classification and 4 node regression tasks.

We measure:
- generalization: how well does the model perform on downstream tasks?
- scalability: what does the performance curve look like with increasing data or compute? (a good foundation model can keep scaling)

We show that naive sequential pretraining yields BAD.
(We show that using [PretrainStrats](paper) yields BETTER.)
We then show that using our improvements, you can get BEST.
Our improvements are:
- Interleaved graph training
  - vs sequential
- Sampling strategies
  - better graph exploration can yield 90% of the downstream performance in 10% of the steps
  - 
- SSL improvements
  - We show that a small extra head (that forces the model to learn the topology) can have a huge boost without hurting other tasks (the representation space is not exhausted).
  
We investigate and explain why our changes yield the best ICL model through:
- graph ladder 
- ssl ladder/experiments
- feature ablations
- graph similarity

We show that our results are valid by getting the same results across multiple models.

Contributions:
- model weights
- 
- contributions on transfer learning