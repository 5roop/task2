# Further investigations

The `lgbt` and `migrants` datasets were merged and shuffled with a predictable random state.

A hyperparameter optimization workflow was setup and it ran, albeit with some weird warnings about input containing `NaN` values sometimes, but after examining the intermediate results on wandb web interface it was determined the model was not being trained correctly. Upon debugging it was determined the error was in the formatting changes introduced after merging the datasets. 

The logs produced with the faulty data had been scrubbed from the W&B website for obvious reasons.

During the optimization some nice examples were noticed, but the bigger part of the runs resulted in abysmally bad performance metrics. The automated result capturing on the W&B website really proved useful for quick sorting and selecting the best hyperparameter setups.

Although the run times were approximately equal to those encountered last time, the search for optimal parameters dragged on and was finally halted after a day. Further investigations will be needed to determine what exactly bayesian search implementation does and how it differs from grid search. It might also be a good idea to introduce all continuous parameter options (e.g. learning rate) as a set of discreet options in order to impose better control over the search options.

About 300 runs were performed in total, afterwards another optimization was started with a narrower scope around the hyperparameters that proved better in the first one. Unfortunately no better setups were found. 

Although I optimized for two hyperparameters with three possibilities each I noticed the optimizer performs more than 9 runs. The reason for this is as of yet unclear to me. After 13 iterations the optimization was interrupted.

## Optimal hyperparameters:

```python
model_args = {
    "num_train_epochs": 10,
    "learning_rate": 0.00002927,
    "train_batch_size": 80
}
```



### Model: `classla/bcms-bertic`

|language|accuracy|f1 score|
|---|---|---|
|hr|0.829|0.82|
|hr|0.836|0.828|
|hr|0.832|0.822|
|hr|0.832|0.823|
|hr|0.835|0.824|
|hr|0.833|0.824|
|hr|0.837|0.827|

Note that the training was performed a few times to get a better picture of its behaviour. May it be noted that in comparison with the results from Task1 these metrics are worse, and the optimization of hyperparameters seems useless, but it must be observed that the input data to both tests were different, as before we only used the `lgbt` dataset and the numbers can not be compared directly.



### Model: `EMBEDDIA/crosloengual-bert`
On Croatian dataset:

|language|accuracy|f1 score|
|---|---|---|
|hr|0.81|0.8|
|hr|0.803|0.792|
|hr|0.8|0.791|
|hr|0.808|0.799|
|hr|0.805|0.795|

On Slovenian dataset:


|language|accuracy|f1 score|
|---|---|---|
|sl|0.757|0.752|
|sl|0.756|0.753|
|sl|0.766|0.761|
|sl|0.758|0.754|
|sl|0.762|0.757|

Slovenian dataset performed significantly better than on previous tests (Task1, same checkpoint), hinting that perhaps the models had been overfit in the past.


All aforementioned results were obtained by training the checkpoint model from scratch. To determine whether repeated training of the same model had any significant effect I also tried that:



|language|model|accuracy|f1 score|
|---|---|---|---|
|hr|classla/bcms-bertic|0.830|0.821|
|hr|classla/bcms-bertic|0.829|0.819|
|hr|classla/bcms-bertic|0.817|0.808|
|hr|classla/bcms-bertic|0.822|0.812|
|hr|classla/bcms-bertic|0.828|0.818|
|hr|classla/bcms-bertic|0.823|0.813|
|hr|classla/bcms-bertic|0.830|0.820|

There seems to be no trend and we got a rough insight into how much training perturbs the performance of the model. 

## On comparing two models:

Supposing we have two models, fine trained on the same training data, we could split test data into multiple ($2\rightarrow 5$) folds and calculate some statistic on either the individual folds, or perhaps even better, on all combinations of folds.

This could be concisely and correctly done using `GroupKFold` from `sklearn.model_selection`.

After acquiring the data it would seem prudent some analysis be done to check whether the t-test can be used, namely if the distribution of the measurements is normal. Since the number of such measurements will likely be small, this is difficult to check, which is why it is probably better to start with Wilcoxon test, which only requires symmetric distribution about its mean value and behaves better for small sample sizes. In "How to avoid machine learning pitfalls: a guide for academic researchers" Michael A. Lones recommends Mann-Whitney's U test for similar reasons. Wilcoxon test expects two related paired samples, which is not the case in our use case, but should be OK anyway.

It would be interesting to check how all tests perform on the same model.

## HuggingFace API for fine tuning

Once more I hit a brickwall when trying to fine tune preexisting models via the HuggingFace interface. The failed attempt is documented in `4-HF trial.ipynb`. Traceback reported being out of memory:

```python
RuntimeError: CUDA out of memory. Tried to allocate 120.00 MiB (GPU 0; 31.75 GiB total capacity; 30.30 GiB already allocated; 92.75 MiB free; 30.46 GiB reserved in total by PyTorch)
```

Inspection with `nvidia-smi` really showed that a lot of resources had been reserved by a process with a weird PID, so I killed all my processes and attempted the training again. Before commencing a `nvidia-smi` command was issued again and showed that no memory had been used up, as shown below:

```
Fri Aug 20 12:26:49 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:03:00.0 Off |                    0 |
| N/A   31C    P0    41W / 300W |      0MiB / 32510MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

but after restarting my training pipeline the same `RuntimeError` was raised. This is a nasty issue, especially because there really should be enough resources available for the allocation of the 120 Mb. Since training without GPU support has proven orders of magnitude more time consuming I shall not pursue that road any more.


## TODO


~~0. Appending the two domain specific datasources to create a single dataset~~

~~1. Hyperparameter optimization~~

2. evaluate the most promising models (per language) on the lgbt+migrants FRENK data

3. perform the evaluation by fine-tuning a model five times (suggestions to more or less iterations welcome), and present the mean of the macro-F1 and accuracy, as well as calculate whether the differences to other models results are statistically significant, quite probably via a t-test (other suggestions welcome, wilcoxon might be better due to small number of observations, or not? - please investigate)

~~4. register with HuggingFace so that you can publish models there~~

~~5. request access to the classla organization at HuggingFace~~

6. publish the best-performing fine-tuned model (so cherry-picked model with best evaluation results among the five performed runs),  with the README / model card containing the evaluation and comparison to alternative models
