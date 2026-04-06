# Figure Captions

## Figure 1: Example orientation tuning curves with von Mises fits

Six example neurons spanning the range of fit quality (R-squared from ~0.31 to ~0.97). Blue circles show mean empirical response per 5-degree orientation bin (+/- SEM). Orange curve shows the fitted von Mises model r(theta) = b + a * exp(kappa * cos(2*(theta - theta0))). Each panel lists the fitted R-squared, kappa, and preferred orientation.

## Figure 2: Distributions of tuning parameters

(Left) Distribution of fitted preferred orientations across all 23,589 neurons. The red dashed line indicates a uniform distribution. Preferred orientations are approximately uniformly distributed, consistent with the lack of columnar orientation maps in mouse V1. (Right) Distribution of fitted tuning sharpness (kappa). The black dashed line indicates the median (kappa = 3.83). The distribution is right-skewed, with most neurons moderately tuned and a tail of very sharply tuned neurons.

## Figure 3: Relationship between tuning sharpness and reliability

(Left) Hexbin density plot of tuning sharpness (kappa) vs. split-half reliability for all 23,589 neurons. Black line shows OLS regression controlling for mean response. Text inset reports Spearman correlation with 95% bootstrap CI. (Right) Mean reliability (+/- SEM) for neurons binned by kappa, showing the monotonic positive trend.

## Figure 4: Population decoder performance vs. neuron count

Mean circular MAE (degrees) of the OLS linear decoder as a function of the number of neurons used, evaluated with 5-fold cross-validation. Error bars show +/- 1 SD across 20 random neuron subsamples (except at 1000 neurons, which uses all available). Orange dashed line: shuffle control MAE (~45 deg). Gray dotted line: theoretical chance level (45 deg). X-axis is on a log scale.

## Figure 5: Decoder predicted vs. true orientation

Scatter plot of predicted vs. true stimulus orientation for the 1000-neuron OLS decoder (80/20 train/test split). Each point is one held-out trial. The tight clustering around the identity line (red dashed) reflects the decoder's high accuracy (MAE = 2.8 deg).
