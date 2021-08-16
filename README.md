# Further investigations



## TODO

0. Hyperparameter optimization 

1. evaluate the most promising models (per language) on the lgbt+migrants FRENK data

2. perform the evaluation by fine-tuning a model five times (suggestions to more or less iterations welcome), and present the mean of the macro-F1 and accuracy, as well as calculate whether the differences to other models results are statistically significant, quite probably via a t-test (other suggestions welcome, wilcoxon might be better due to small number of observations, or not? - please investigate)

3. register with HuggingFace so that you can publish models there

4. request access to the classla organization at HuggingFace

5. publish the best-performing fine-tuned model (so cherry-picked model with best evaluation results among the five performed runs),  with the README / model card containing the evaluation and comparison to alternative models
