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



### Model: EMBEDDIA/crosloengual-bert

|language|accuracy|f1 score|
|---|---|---|
|hr|0.81|0.8|
|hr|0.803|0.792|
|hr|0.8|0.791|
|hr|0.808|0.799|
|hr|0.805|0.795|


    

## On comparing two models:

Supposing we have two models, fine trained on the same training data, we could split test data into multiple ($2\rightarrow 5$) folds and calculate some statistic on either the individual folds, or perhaps even better, on all combinations of folds.

This could be concisely and correctly done using `GroupKFold` from `sklearn.model_selection`.

After acquiring the data it would seem prudent some analysis be done to check whether the t-test can be used, namely if the distribution of the measurements is normal. Since the number of such measurements will likely be small, this is difficult to check, which is why it is probably better to start with Wilcoxon test, which only requires symmetric distribution about its mean value and behaves better for small sample sizes.

It would be interesting to check how both tests perform on the same model. 


## TODO


~~0. Appending the two domain specific datasources to create a single dataset~~

~~1. Hyperparameter optimization~~

2. evaluate the most promising models (per language) on the lgbt+migrants FRENK data

3. perform the evaluation by fine-tuning a model five times (suggestions to more or less iterations welcome), and present the mean of the macro-F1 and accuracy, as well as calculate whether the differences to other models results are statistically significant, quite probably via a t-test (other suggestions welcome, wilcoxon might be better due to small number of observations, or not? - please investigate)

~~4. register with HuggingFace so that you can publish models there~~

~~5. request access to the classla organization at HuggingFace~~

6. publish the best-performing fine-tuned model (so cherry-picked model with best evaluation results among the five performed runs),  with the README / model card containing the evaluation and comparison to alternative models
