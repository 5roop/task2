# Further investigations

The `lgbt` and `migrants` datasets were merged and shuffled with a predictable random state.

A hyperparameter optimization workflow was setup and it ran, albeit with some weird warnings about input containing `NaN` values sometimes, but after examining the intermediate results on wandb web interface it was determined the model was not being trained correctly. Upon debugging it was determined the error was in the formatting changes introduced after merging the datasets. 

The logs produced with the faulty data had been scrubbed from the W&B website for obvious reasons.

During the optimization some nice examples were noticed, but the bigger part of the runs resulted in abysmally bad performance metrics. The automated result capturing on the W&B website really proved useful for quick sorting and selecting the best hyperparameter setups.

Although the run times were approximately equal to those encountered last time, the search for optimal parameters dragged on and was finally halted after a day. Further investigations will be needed to determine what exactly bayesian search implementation does and how it differs from grid search. It might also be a good idea to introduce all continuous parameter options (e.g. learning rate) as a set of discreet options in order to impose better control over the search options.

About 300 runs were performed in total, afterwards another optimization was started with a narrower scope around the hyperparameters that proved better in the first one. Unfortunately no better setups were found. 

Although I optimized for two hyperparameters with three possibilities each I noticed the optimizer performs more than 9 runs. The reason for this is unclear to me.



## TODO


~~0. Appending the two domain specific datasources to create a single dataset~~

1. Hyperparameter optimization 



2. evaluate the most promising models (per language) on the lgbt+migrants FRENK data

3. perform the evaluation by fine-tuning a model five times (suggestions to more or less iterations welcome), and present the mean of the macro-F1 and accuracy, as well as calculate whether the differences to other models results are statistically significant, quite probably via a t-test (other suggestions welcome, wilcoxon might be better due to small number of observations, or not? - please investigate)

~~4. register with HuggingFace so that you can publish models there~~

~~5. request access to the classla organization at HuggingFace~~

6. publish the best-performing fine-tuned model (so cherry-picked model with best evaluation results among the five performed runs),  with the README / model card containing the evaluation and comparison to alternative models
