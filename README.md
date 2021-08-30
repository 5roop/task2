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

Finally after clicking through many issues on github and HF forums an answer was found stating that not many devices can handle batch sizes greater than 4 (although this was not an issue with `simpletransformers`...) After changing that parameter it worked.

```Training completed. Do not forget to share your model on huggingface.co/models =)```

Evaluating the model proved a bit more difficult, as HF interface doesn't seem to include a high level predict methods. `simpletransformers` offer that, but saving and uploading the model is not documented in the docs. I therefore opted for a hybrid approach where I trained the model with HF interface, saved it locally, loaded it with `simpletransformers` and performed evaluations there.

My first attempt was dissapointing: after training the model successully in HF I saved it and evaluated it on test data. Without further training I obtained the following results:

|Language | model | method | accuracy | f_1|
| --- | ---| ---| ---| ---|
|hr|classla/bcms-bertic | training: HF, evaluation: simpletransformers | 0.597 |  0.374|

The target accuracy obtained in previous runs was higher than 0.8, so this is quite a miserable result. The hyperparameters used were
```
    output_dir = "./outputs",
    num_train_epochs = 30,
    per_device_train_batch_size = 4,
    warmup_steps = 500,
    learning_rate = 3e-5,
    logging_dir = "./runs",
    overwrite_output_dir=True
```

I tried increasing the number of training epochs to 100 to compensate for the lowered batchsize, but I encoutered errors that rendered this option unfeasible, so I settled for 30. Sadly, the results are even worse.

|Language | model | method | accuracy | f_1|
| --- | ---| ---| ---| ---|
|hr|classla/bcms-bertic | training: HF, evaluation: simpletransformers | 0.429 |  0.406|

Further attempts at optimizing the setup raised fatal errors and produced so much auxiliary data that the disk was soon full and regular flushing was required to mitigate that. After decreasing the number of epochs to a managable amount the performance dropped even more.


All the problems mentioned above mean that it will probably be necessary to find a way to export models from `simpletransformers` to HF and then publish them.

I returned to `simpletransformers` and trained the model as before. I got familiar with the parameters that  allowed me control over the output destination. As it turned out just specifying the output directory as the checkpoint is enough for HF to load the tokenizer and the model, but using the loaded model proved difficult as the tokenizer could not extract all the necessary parameters from the given file.


After carefully reviewing my HF code I discovered a bug in it and after correcting it I trained a HF model again. More fiddling was necessary to prevent errors due to the lack of disk space. Finally I was able to produce a model that on the first evaluation achieved accuracies and f1 scores of about 0.8, which is acceptable. Subsequent evaluations however fluctuated a great deal. I compiled a table below:


### Model: ./finetuned_models/HR_hate___classla_bcms-bertic_5/

|language|accuracy|f1 score|
|---|---|---|
|hr|0.7|0.699|
|hr|0.559|0.3782|
|hr|0.393|0.337|
|hr|0.808|0.798|
|hr|0.19|0.1880|
|hr|0.217|0.2024|
|hr|0.218|0.217|
|hr|0.418|0.351|
|hr|0.758|0.758|
|hr|0.188|0.1880|
|hr|0.391|0.280|
|hr|0.494|0.456|
|hr|0.272|0.2265|
|hr|0.681|0.632|
|hr|0.798|0.7|
|hr|0.198|0.1976|
|hr|0.29|0.2898|
|hr|0.811|0.799|
|hr|0.345|0.334|
|hr|0.754|0.734|

When evaluating it with kernel restarts between evaluations the situation did not improve significantly:

|language|accuracy|f1 score|
|---|---|---|
|hr|0.775|0.768|
|hr|0.508|0.495|
|hr|0.646|0.566|
|hr|0.186|0.184|
The issue seems to stem from randomly initiated layers in the `BertForSequenceClassification` model, indicating that training the model in HF is not enough on its own, and even pretrained checkpoints should be trained in `simpletransformers` as well.

After implementing this methodology I finally get consistent performance:
|language|accuracy|f1 score|
|---|---|---|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|
|hr|0.815|0.806|

In order to evaluate different models it will therefore be necessary to train and evaluate the models in subsequent runs. I proceeded with evaluating my prevously pretrained checkpoint and obtained the following results:

|language|accuracy|f1 score|
|---|---|---|
|hr|0.811|0.801|
|hr|0.811|0.802|
|hr|0.819|0.81|
|hr|0.821|0.811|
|hr|0.82|0.810|
|hr|0.817|0.808|
|hr|0.818|0.808|
|hr|0.817|0.807|
|hr|0.817|0.808|
|hr|0.815|0.804|

Upon using the same methodology for stock `classla/bcms-bertic ` checkpoint, I obtained the following statistics:
|language|accuracy|f1 score|
|---|---|---|
|hr|0.832|0.823|
|hr|0.833|0.825|
|hr|0.831|0.821|
|hr|0.827|0.817|
|hr|0.83|0.82|
|hr|0.829|0.82|
|hr|0.832|0.823|
|hr|0.834|0.824|
|hr|0.832|0.824|
|hr|0.833|0.824|

It is unfortunately very clear that we did not manage to best the already published checkpoint on the HF model hub. I trained the saved checkpoint some more (for another 5 epochs, as apparently the virtual machine can not handle more than that). After this I was able to achieve marginally better results, albeit still worse than what I can do with `simpletransformers` in a fraction of the time. Results are attached below:

|language|accuracy|f1 score|
|---|---|---|
    |hr|0.824|0.814|
|hr|0.825|0.816|
|hr|0.823|0.814|
|hr|0.823|0.812|
|hr|0.825|0.815|
|hr|0.823|0.814|
|hr|0.821|0.811|
|hr|0.824|0.815|
|hr|0.82|0.809|
|hr|0.822|0.812|

Since the performance is consistantly better, I decide to repeat the training with HF for a few times. Unfortunately, the results were not much better after 5 further iterations:
|language|accuracy|f1 score|
|---|---|---|
|hr|0.812|0.803|
|hr|0.812|0.804|
|hr|0.815|0.806|
|hr|0.815|0.805|
|hr|0.808|0.801|
|hr|0.814|0.806|
|hr|0.809|0.803|
|hr|0.811|0.803|
|hr|0.809|0.802|
|hr|0.81|0.804|

I kept the successive finetuned models and compared also the middle stages, but they achieved similar results than the result above and still couldn't surpass the performance we saw with just one training in `simpletransformers`.

With this abnoxious detail in mind I decided not to pursue the final stage, which would be uploading the model to HuggingFace model hub. Although the training took quite some time I found it even more annoying that the evaluation phase needed so much optimization before predictions could be made. In the future before receiving specific hints about possible improvements I wanted to pursue two pathways:

* Reduce the optimization parameters in the evaluation phase so that the evaluation is performed quicker and check if the results differ significantly (so if even with pruned training the published version is better than my 'finetuned' checkpoint)
* Check whether some other published model checkpoint might benefit from additional training.

I opted for the latter bulletpoint as it is more honest and scientifically justifiable than the first one. One of the models that also proved quite good in the previous tests was `crosloengual-bert`, so I left it overnight to train 5 times (about 10 hours of wall time), after each iteration I ran a command that purged the auxiliary files to prevent errors due to low disk space, and in the morning discovered the same trend:

### Model: ./finetuned_models/HR_hate___EMBEDDIA/crosloengual-bert_5

|language|accuracy|f1 score|
|---|---|---|
|hr|0.74|0.728|
|hr|0.74|0.728|
|hr|0.745|0.731|
|hr|0.746|0.733|
|hr|0.739|0.726|
|hr|0.743|0.73|
|hr|0.742|0.727|
|hr|0.744|0.728|
|hr|0.74|0.725|
|hr|0.739|0.724|

### Model: EMBEDDIA/crosloengual-bert

|language|accuracy|f1 score|
|---|---|---|
|hr|0.806|0.798|
|hr|0.798|0.789|
|hr|0.798|0.789|
|hr|0.805|0.796|
|hr|0.805|0.796|
|hr|0.805|0.796|
|hr|0.808|0.8|
|hr|0.808|0.798|
|hr|0.809|0.8|
|hr|0.806|0.797|

It is again clear that we do not need sophisticated statistical tools to determine to determine that our model is not yet worthy of publication. Judging from the trend observed not even longer training times can improve the accuracies. 


## Technical details

Training with HF was performed 5 times with the following parameters:

```python
training_args = TrainingArguments(
    output_dir = "./outputs",
    num_train_epochs = 7,
    per_device_train_batch_size = 4,
    warmup_steps = 100,
    learning_rate = 3e-5,
    logging_dir = "./runs",
    overwrite_output_dir=True
)
```

and when evaluating, `simpletransformers` training was used with these parameters:
```python
model_args = {
        "num_train_epochs": 5,
        "learning_rate": 1e-5,
        "overwrite_output_dir": True,
        "train_batch_size": 40
    }
```

## TD;DR

* I perform training with HF and evaluation (which requires some further training) with `simpletransformers`
* HF crashes unexpectedly if the parameters are not carefully optimized and produces a lot of data in its wake.
* Models are initialized with some degree of randomness which renders the pretrained models useless if some training is not performed on them upon loading.
* Finetuning does not seem to improve the statistics.
* The methodology for comparing two models is ready; due to non-deterministic behaviour of loaded models they can be pretrained and then evaluated, yielding a measurement which can be recorded and, after gathering a decent sample, analyzed.
## TODO


~~0. Appending the two domain specific datasources to create a single dataset~~

~~1. Hyperparameter optimization~~

~~2. evaluate the most promising models (per language) on the lgbt+migrants FRENK data~~

~~3. perform the evaluation by fine-tuning a model five times (suggestions to more or less iterations welcome), and present the mean of the macro-F1 and accuracy, as well as calculate whether the differences to other models results are statistically significant, quite probably via a t-test (other suggestions welcome, wilcoxon might be better due to small number of observations, or not? - please investigate)~~

~~4. register with HuggingFace so that you can publish models there~~

~~5. request access to the classla organization at HuggingFace~~

6. publish the best-performing fine-tuned model (so cherry-picked model with best evaluation results among the five performed runs),  with the README / model card containing the evaluation and comparison to alternative models
