# Further investigations



## TODO


0. Appending the two domain specific datasources to create a single dataset
1. Hyperparameter optimization 



2. evaluate the most promising models (per language) on the lgbt+migrants FRENK data

3. perform the evaluation by fine-tuning a model five times (suggestions to more or less iterations welcome), and present the mean of the macro-F1 and accuracy, as well as calculate whether the differences to other models results are statistically significant, quite probably via a t-test (other suggestions welcome, wilcoxon might be better due to small number of observations, or not? - please investigate)

4. register with HuggingFace so that you can publish models there

5. request access to the classla organization at HuggingFace

6. publish the best-performing fine-tuned model (so cherry-picked model with best evaluation results among the five performed runs),  with the README / model card containing the evaluation and comparison to alternative models
