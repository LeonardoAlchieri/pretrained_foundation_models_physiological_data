# Appendix

## A Detailed Results

We present, in Table C and Table D, detailed results for the LOPO and 5-fold TAcv validation methods, respectively. The additional metrics reported are: F1-score, precision and the area under the receiving operating curve (ROC AUC). We also report results for a different set of weights for the Chronos foundation model, which we call Chronos Small.

## B Reproducibility Information

**Handcrafted Feature Set.** In Table E we report a detailed description of our handcrafted features. The feature-set is the same as [4] and computed for each EDA component, i.e., phasic, tonic and “mixed”.

**Hyperparameter tuning.** We perform linear probing using a logistic regression classifier for our experiments, as described in Section 4. When training, we perform hyperparameter tuning using a grid search 3-fold cross validation. We tune the regularization factor by trying the following values: λ = [0.01, 0.1, 1, 10], where λ is the inverse of the regularization factor. We set the solver to `lbfgs`, we use L2 regularization and we set the maximum number of iterations to 10 000.

Using the same method, we also select the baseline classifier. We use a 3-fold cross validation over the train set for the following baseline classifiers: _most frequent_, which always predicts the majority class according to the training set; _stratified_, also known as biased random guess, which predicts the label distribution from the training set; _uniform_, which predicts the labels using a uniform distribution.

**Embeddings concatenation.** As we report in Section 4, to obtain our results we average the foundation model features along the EDA components. We report here two additional strategies that we tested: time averaging, where we average across the time axis, i.e.,  
$$
\mathbf{e} \;\to\; \frac{1}{T}\sum_{t=1}^T \mathbf{e}_t
$$  
and a simple concatenation, where we just concatenate across all dimensions. We compare the performance of these two strategies, along with the channel-averaging strategy, on the USILaughs only in ??. Our results show that no significant difference is present between the three strategies.

**Foundation models.** We use pre-trained weights available as open source for all three of the foundation models. We report in Table A the link to both the repository with the code and the model weights for all three foundation models.

---

### Table A: Links to foundation model repositories and pretrained weights

| Model       | GitHub Repository                                    | Pretrained Weights                                                      |
|-------------|-------------------------------------------------------|-------------------------------------------------------------------------|
| Chronos     | amazon-science/chronos-forecasting                    | chronos-t5-small (small)<br/>chronos-t5-large (large)                   |
| MOMENT      | moment-timeseries-foundation-model/moment             | MOMENT-1-large (large)                                                  |
| TSMixer     | ditschuk/pytorch-tsmixer                             | granite-timeseries-patchtsmixer                                         |

---

### Table B: Results on the USILaughs dataset grouped by feature set and aggregation strategy (LOPO validation)

| Feature Set       | Aggregation Strategy | Balanced Accuracy | MCC            | Recall         |
|-------------------|----------------------|-------------------|----------------|----------------|
| **Chronos Large** | Channel Averaging    | .72 ± .07         | .48 ± .15      | .66 ± .11      |
|                   | Concatenation        | .73 ± .07         | .48 ± .15      | .67 ± .10      |
|                   | Time Averaging       | .76 ± .07         | .53 ± .15      | .74 ± .10      |
| **Chronos Small** | Channel Averaging    | .72 ± .08         | .46 ± .17      | .65 ± .11      |
|                   | Concatenation        | .74 ± .08         | .50 ± .18      | .69 ± .11      |
|                   | Time Averaging       | .73 ± .09         | .47 ± .19      | .70 ± .12      |
| **MOMENT Large**  | Channel Averaging    | .55 ± .04         | .12 ± .11      | .34 ± .07      |
|                   | Concatenation        | .54 ± .05         | .08 ± .10      | .42 ± .07      |
|                   | Time Averaging       | .54 ± .05         | .08 ± .10      | .42 ± .07      |
| **Granite-TSMixer** | Channel Averaging  | .71 ± .12         | .43 ± .24      | .73 ± .12      |
|                   | Concatenation        | .69 ± .11         | .41 ± .22      | .73 ± .10      |
|                   | Time Averaging       | .69 ± .11         | .41 ± .22      | .73 ± .10      |

---

### Table C: Additional results for the LOPO validation method

| Task (Dataset)               | Feature Set         | Balanced Accuracy | F1             | MCC            | Precision      | Recall         | ROC AUC         |
|------------------------------|---------------------|-------------------|----------------|----------------|----------------|----------------|-----------------|
| **Low/High Engagement**<br/>(APSync) | Chronos Large      | .50 ± .00         | .07 ± .13      | -.02 ± .03     | .04 ± .08      | .14 ± .28      | .50 ± .00       |
|                              | Chronos Small      | .50 ± .05         | .30 ± .19      | -.00 ± .10     | .31 ± .21      | .31 ± .27      | .50 ± .05       |
|                              | MOMENT Large       | .50 ± .00         | .07 ± .13      | .00 ± .00      | .04 ± .08      | .14 ± .28      | .50 ± .00       |
|                              | Granite-TSMixer     | .49 ± .03         | .29 ± .10      | -.01 ± .08     | .47 ± .20      | .36 ± .23      | .49 ± .03       |
|                              | Handcrafted        | .64 ± .09         | .54 ± .16      | .28 ± .16      | .62 ± .17      | .58 ± .25      | .64 ± .09       |
|                              | Random Baseline    | .52 ± .03         | .32 ± .17      | .03 ± .05      | .27 ± .17      | .46 ± .27      | .52 ± .03       |
| **Sleep/Wake**<br/>(BiHeartS)       | Chronos Large      | .75 ± .05         | .80 ± .03      | .37 ± .09      | .92 ± .05      | .71 ± .04      | .75 ± .05       |
|                              | Chronos Small      | .69 ± .04         | .76 ± .04      | .27 ± .05      | .89 ± .08      | .66 ± .03      | .69 ± .04       |
|                              | MOMENT Large       | .62 ± .05         | .77 ± .05      | .20 ± .10      | .90 ± .05      | .69 ± .07      | .62 ± .05       |
|                              | Granite-TSMixer     | .58 ± .02         | .71 ± .05      | .12 ± .04      | .85 ± .06      | .61 ± .05      | .58 ± .02       |
|                              | Handcrafted        | .75 ± .07         | .80 ± .03      | .37 ± .12      | .93 ± .05      | .71 ± .05      | .75 ± .07       |
|                              | Random Baseline    | .48 ± .04         | .43 ± .22      | -.03 ± .05     | .55 ± .29      | .36 ± .18      | .48 ± .04       |
| **Low/High Engagement**<br/>(SEED)      | Chronos Large      | .63 ± .13         | .73 ± .12      | -.00 ± .02     | .66 ± .13      | .89 ± .12      | .50 ± .00       |
|                              | Chronos Small      | .63 ± .12         | .73 ± .12      | .00 ± .01      | .66 ± .13      | .90 ± .12      | .50 ± .00       |
|                              | MOMENT Large       | .63 ± .13         | .73 ± .12      | -.01 ± .01     | .66 ± .13      | .90 ± .12      | .50 ± .00       |
|                              | Granite-TSMixer     | .63 ± .13         | .74 ± .12      | .00 ± .00      | .66 ± .13      | .91 ± .12      | .50 ± .00       |
|                              | Handcrafted        | .62 ± .12         | .73 ± .12      | -.01 ± .02     | .66 ± .13      | .90 ± .12      | .50 ± .00       |
|                              | Random Baseline    | .56 ± .06         | .61 ± .10      | .01 ± .01      | .66 ± .13      | .64 ± .12      | .51 ± .01       |
| **Cognitive Load/Relaxation**<br/>(USILaughs) | Chronos Large      | .72 ± .07         | .66 ± .09      | .48 ± .15      | .77 ± .10      | .66 ± .11      | .72 ± .07       |
|                              | Chronos Small      | .72 ± .08         | .66 ± .11      | .46 ± .17      | .73 ± .12      | .65 ± .11      | .72 ± .08       |
|                              | MOMENT Large       | .55 ± .04         | .40 ± .07      | .12 ± .11      | .57 ± .11      | .34 ± .07      | .55 ± .04       |
|                              | Granite-TSMixer     | .71 ± .12         | .70 ± .12      | .43 ± .24      | .75 ± .13      | .73 ± .12      | .71 ± .12       |
|                              | Handcrafted        | .73 ± .09         | .74 ± .09      | .47 ± .20      | .75 ± .11      | .78 ± .08      | .73 ± .09       |
|                              | Random Baseline    | .51 ± .03         | .40 ± .06      | .01 ± .07      | .40 ± .06      | .41 ± .07      | .51 ± .03       |

---

### Table D: Results for the 5-fold TAcv method

| Task (Dataset)               | Feature Set         | Balanced Accuracy | F1             | MCC            | Precision      | Recall         | ROC AUC         |
|------------------------------|---------------------|-------------------|----------------|----------------|----------------|----------------|-----------------|
| **Low/High Engagement**<br/>(APSync) | Chronos Large      | .50 ± .00         | .00 ± .00      | .00 ± .00      | .00 ± .00      | .00 ± .00      | .50 ± .00       |
|                              | Chronos Small      | .52 ± .05         | .30 ± .21      | .04 ± .11      | .36 ± .20      | .31 ± .27      | .52 ± .05       |
|                              | MOMENT Large       | .50 ± .00         | .00 ± .00      | .00 ± .00      | .00 ± .00      | .00 ± .00      | .50 ± .00       |
|                              | Granite-TSMixer     | .52 ± .05         | .36 ± .19      | .02 ± .13      | .41 ± .24      | .37 ± .21      | .52 ± .05       |
|                              | Handcrafted        | .62 ± .14         | .46 ± .30      | .24 ± .29      | .63 ± .36      | .47 ± .33      | .62 ± .14       |
|                              | Random Baseline    | .50 ± .03         | .39 ± .20      | .00 ± .06      | .36 ± .20      | .44 ± .22      | .50 ± .03       |
| **Sleep/Wake**<br/>(BiHeartS)       | MOMENT Large       | .62 ± .05         | .77 ± .05      | .20 ± .10      | .88 ± .05      | .70 ± .08      | .62 ± .05       |
|                              | Chronos Large      | .75 ± .05         | .80 ± .03      | .37 ± .09      | .92 ± .05      | .71 ± .04      | .75 ± .05       |
|                              | Chronos Small      | .69 ± .04         | .76 ± .02      | .27 ± .05      | .89 ± .07      | .66 ± .03      | .69 ± .04       |
|                              | Granite-TSMixer     | .61 ± .05         | .71 ± .05      | .12 ± .04      | .85 ± .06      | .61 ± .05      | .61 ± .05       |
|                              | Handcrafted        | .75 ± .07         | .80 ± .03      | .37 ± .12      | .93 ± .05      | .71 ± .05      | .75 ± .07       |
|                              | Random Baseline    | .48 ± .04         | .43 ± .22      | -.03 ± .05     | .55 ± .29      | .36 ± .18      | .48 ± .04       |
| **Low/High Engagement**<br/>(SEED)      | Chronos Large      | .50 ± .00         | .79 ± .07      | -.00 ± .01     | .67 ± .07      | .96 ± .07      | .50 ± .00       |
|                              | Chronos Small      | .50 ± .01         | .79 ± .05      | -.01 ± .02     | .67 ± .07      | .97 ± .07      | .50 ± .01       |
|                              | MOMENT Large       | .49 ± .01         | .78 ± .07      | -.04 ± .04     | .67 ± .08      | .93 ± .06      | .49 ± .01       |
|                              | Granite-TSMixer     | .50 ± .00         | .80 ± .05      | .00 ± .00      | .68 ± .07      | 1.00 ± .00     | .50 ± .00       |
|                              | Handcrafted        | .52 ± .05         | .81 ± .06      | .08 ± .18      | .68 ± .09      | 1.00 ± .00     | .52 ± .05       |
|                              | Random Baseline    | .49 ± .03         | .65 ± .13      | -.02 ± .05     | .66 ± .09      | .69 ± .25      | .49 ± .03       |
| **Cognitive Load/Relaxation**<br/>(USILaughs) | Chronos Large      | .77 ± .09         | .74 ± .10      | .56 ± .18      | .78 ± .11      | .73 ± .16      | .77 ± .09       |
|                              | Chronos Small      | .78 ± .07         | .75 ± .09      | .58 ± .13      | .77 ± .07      | .77 ± .17      | .78 ± .07       |
|                              | MOMENT Large       | .58 ± .04         | .46 ± .06      | .18 ± .10      | .59 ± .10      | .39 ± .08      | .58 ± .04       |
|                              | Granite-TSMixer     | .75 ± .09         | .75 ± .08      | .51 ± .17      | .67 ± .09      | .84 ± .08      | .75 ± .09       |
|                              | Handcrafted        | .75 ± .07         | .75 ± .05      | .52 ± .12      | .67 ± .09      | .88 ± .03      | .75 ± .07       |
|                              | Random Baseline    | .50 ± .05         | .29 ± .24      | .00 ± .11      | .27 ± .23      | .31 ± .25      | .50 ± .05       |

---

### Table E: Mathematical formulas for the 9 computed hand-crafted time-domain features, the 2 EDA-specific features and the 4 frequency-domain features

We compute these features for the three EDA components, i.e., tonic, phasic and “mixed”, for a total of 45 features.

| Feature                                  | Mathematical Notation or Formula                                                                                  |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Time domain**                          |                                                                                                                    |
| Mean                                     | \( \displaystyle \frac{1}{N}\sum_{i=1}^N x_i \)                                                                     |
| Minimum                                  | \( \displaystyle \min(x_1, x_2, \dots, x_N) \)                                                                     |
| Maximum                                  | \( \displaystyle \max(x_1, x_2, \dots, x_N) \)                                                                     |
| Standard Deviation                       | \( \displaystyle \sqrt{\frac{1}{N-1}\sum_{i=1}^N \bigl(x_i - \bar{x}\bigr)^2} \)                                    |
| Dynamic Range                            | \( \displaystyle \max(x_1, \dots, x_N) - \min(x_1, \dots, x_N) \)                                                   |
| Slope                                    | \( \displaystyle \frac{x_N - x_1}{N - 1} \)                                                                         |
| Absolute Value of Slope                  | \( \displaystyle \biggl|\frac{x_N - x_1}{N - 1}\biggr| \)                                                           |
| Mean of the First Derivative            | \( \displaystyle \frac{1}{N-1}\sum_{i=1}^{N-1} (x_{i+1} - x_i) \)                                                    |
| Standard Deviation of the First Derivative | \( \displaystyle \sqrt{\frac{1}{N-2}\sum_{i=1}^{N-1}\bigl((x_{i+1}-x_i) - \overline{(x_{i+1}-x_i)}\bigr)^2} \)      |
| **EDA-specific features**                |                                                                                                                    |
| Number of EDA Peaks in a Window         | Count of local maxima in the windowed EDA signal                                                                   |
| Amplitude of EDA Peaks                   | Amplitude of local maxima in the windowed EDA signal                                                               |
| **Frequency domain** (Fast Fourier Transform) |                                                                                                                |
| Direct Current                           | \( X_0 \)                                                                                                          |
| Sum of frequency coefficients           | \( \displaystyle \sum_{k=1}^N \lvert X_k\rvert \)                                                                  |
| Information entropy                      | \( \displaystyle \sum_{k=1}^N P(X_k)\,\log_2\bigl(P(X_k)\bigr) \)                                                   |
| Spectral energy                          | \( \displaystyle \sum_{k=1}^N \lvert X_k\rvert^2 \)                                                                |

:contentReference[oaicite:1]{index=1}
