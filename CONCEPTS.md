
## "Extract frozen AST audio + R(2+1)D-18 video embeddings for Urbansas..."

In this experiment, **extracting** means passing all 1,080 Urbansas audio and video clips through two already-trained neural networks and collecting the numerical outputs from their internal feature layers. Specifically, it pass each audio clip through the AST encoder and collect a 768-dimensional vector. it pass each video clip through the R(2+1)D-18 network and collect a 512-dimensional vector. These vectors are called embeddings. it then save them as `.npy` files so that it can reuse them without processing the original clips again.

**Frozen** means that the internal weights of AST and R(2+1)D-18 are not updated while processing this Urbansas dataset. AST was pretrained on AudioSet, and R(2+1)D-18 was pretrained on Kinetics-400. it's using the knowledge they already learned rather than training those large networks again on this relatively small Urbansas dataset. In the extraction script, `torch.no_grad()` prevents gradient calculations, and `model.eval()` puts the networks into evaluation mode. These choices make extraction faster and avoid updating the encoders.

**Why freeze them?** it want to isolate one particular research question: whether combining the representations learned by two existing pretrained models can improve incident prediction. If it fine-tuned both encoders on Urbansas, their representations would change during training, introducing another variable. Freezing them keeps the feature extractors fixed, making the fusion experiment easier to interpret and less computationally expensive. It also reduces the risk of overfitting a relatively small dataset, although freezing alone does not guarantee freedom from data leakage.

**Why extract now?** In Week 9c, it combined the final audio and video probabilities. That experiment used only two numbers per clip, so much of the information in the original representations was already compressed. Now, in Milestone C, it want to combine the richer internal representations instead. it therefore need to extract and save those embeddings before training the new fusion MLP. Once extracted, C2/(next week's work) can train and evaluate fusion models using the saved vectors instead of repeatedly running the large audio and video networks.

**The distinction is:** it's not extracting the models themselves, and it's not extracting their weights. it's extracting the numerical feature representations they produce when they process this Urbansas clips. The encoders are frozen because it want those representations to remain unchanged while it investigate whether a new trainable MLP can combine them more effectively.

---

**Systematically offset:** Audio and video embeddings have different value distributions. Audio embeddings are centered near zero (mean 0.0185), while video embeddings have a positive mean (0.7062). This means the video features are systematically shifted relative to the audio features. Standardization helps bring each feature to a comparable scale by centering it around zero and scaling it to unit variance.

**LOLA folds:** Leave-One-Location-Out is an evaluation method where the model is trained on four locations and tested on the fifth, repeating this process until every location has been held out once. This tests whether the model can generalize to an entirely unseen location rather than simply recognize patterns from locations it has already encountered.

**Per-fold:** This means performing an operation separately within each evaluation fold. For example, the StandardScaler is fitted using only the four training locations, then applied to both training and test data. This prevents information from the held-out location from leaking into the training process.

**Audio-only MLP vs. Feature-fusion MLP:** The audio-only MLP takes 768 audio features as input, passes them through a 256-neuron hidden layer, and predicts two classes. The feature-fusion MLP takes 1,280 combined audio and video features (768 + 512), uses the same 256-neuron hidden layer, and predicts the same two classes. Comparing them helps determine whether adding video features improves performance beyond what a trainable audio-only model can achieve.

**Why the control matters:** The audio-only MLP is the control experiment, while the feature-fusion MLP is the multimodal experiment. If feature fusion performs better than the audio-only MLP, it provides evidence that video contributes useful complementary information. If both perform similarly, the addition of video has not demonstrated a meaningful improvement.


---
# Week 10 — Four lessons, with the reasons behind them

## Lesson 1 — More complex models aren't automatically more trustworthy.

**What we learned:**

Our frozen-encoder AST audio model had better calibration on MAVD than the feature-fusion MLP models.

| Model                     | MAVD ECE before scaling |
| ------------------------- | ----------------------- |
| AST audio                 | 0.158                   |
| Feature-fusion audio-only | 0.201                   |
| Feature-fusion video-only | 0.274                   |
| Feature-fusion MLP        | 0.280                   |

**Why did this happen?**

The AST model uses pretrained audio features with a relatively simple prediction head. The feature-fusion MLPs have additional trainable parameters and can learn more complicated relationships.

That extra flexibility can help a model learn useful patterns, but it can also make the model produce probabilities that are more extreme than its actual predictive performance justifies.

For example:

- Model A predicts 90% confidence and is correct 70% of the time.
- Model B predicts 75% confidence and is correct 73% of the time.

Model A sounds more confident, but Model B gives more trustworthy probabilities.

Temperature scaling substantially improved some MLPs, which showed that their original confidence estimates could be corrected without retraining the underlying models.

Why does this lesson matter?

When building ML systems, we shouldn't assume that a more complicated architecture produces more reliable predictions.

A model needs to be evaluated on multiple dimensions:

- Discrimination: Can it distinguish positive from negative examples?
- Calibration: Do its probabilities reflect reality?
- Generalization: Does it work on data it hasn't seen?

A sophisticated model that produces misleading confidence estimates can be dangerous in a decision-making system.

## Lesson 2 — Calibration learned on one dataset doesn't automatically transfer to another.

**What we learned:**

AST audio had relatively good calibration on **MAVD**:

\- MAVD ECE: 0.158

But on the external TAU dataset:

\- TAU ECE: 0.328

We then fitted temperature scaling using MAVD predictions and applied the learned temperature to TAU.

The improvements on TAU were very small.

**Why did this happen?**

Because MAVD and TAU are different data distributions.

For example, their positive-class prevalence differs:

| Dataset | Positive examples |
| ------- | ----------------- |
| MAVD    | 25.6%             |
| TAU     | 15.0%             |

The models learned their representations and prediction behavior using MAVD data. When they encountered TAU, the relationship between their predictions and actual outcomes changed.

Think of it this way:

You train a model to recognize traffic incidents in one group of cities. It learns patterns from those cities.

Then you move it to another group of cities, where traffic conditions, background sounds, recording environments, and incident frequencies differ.

Even if the model recognizes some useful patterns, its confidence estimates may no longer mean the same thing.

**Why does this lesson matter?**

Real-world ML systems rarely operate forever on exactly the same distribution they were trained on.

They encounter:

- New locations.
- Different sensors.
- Different populations.
- Changing environmental conditions.
- Different class distributions.

Therefore, good performance on the original dataset is not sufficient evidence of reliability in a new environment.

This is why external validation and dataset-shift analysis are important parts of production ML engineering.

## Lesson 3 — Ranking and calibration are two different capabilities.

**What we learned:**

On MAVD, feature-fusion audio-only had better discrimination than AST audio:

| Model                     | ROC-AUC | ECE before scaling |
| ------------------------- | ------- | ------------------ |
| AST audio                 | 0.6801  | 0.158              |
| Feature-fusion audio-only | 0.7042  | 0.201              |

AST audio had better calibration, but feature-fusion audio-only ranked positive examples above negative examples more effectively.

After temperature scaling, feature-fusion audio-only ECE improved to 0.107.

Why are these different?

Imagine two models evaluating 100 clips.

Model A:

- Correctly ranks most positive clips above negative clips.
- But gives excessively high probabilities.

Model B:

- Produces probabilities that match observed frequencies.
- But struggles to rank positive clips above negative clips.

Model A has better discrimination.

Model B may have better calibration.

The distinction is:

- Discrimination: Does the model assign higher scores to positive examples than negative examples?
- Calibration: When the model predicts a certain probability, does that probability match the observed frequency?

A model can have excellent ranking but poor calibration, or good calibration but weak ranking.

Temperature scaling helped improve the confidence estimates of feature-fusion audio-only without changing its ranking.

Why does this lesson matter?

Different applications need different capabilities.

For example:

- Incident retrieval needs strong discrimination: find the most likely incidents first.
- Risk estimation needs trustworthy probabilities: understand how likely an incident actually is.
- Automated response systems need both: identify incidents and make decisions using reliable probabilities.

We shouldn't use ROC-AUC alone to claim that a model is ready for deployment, and we shouldn't use ECE alone either.

## Lesson 4 — A good ECE score doesn't automatically mean useful confidence.

What we learned:

The fine-tuned video model on TAU had the lowest ECE:

Lowest measured ECE

# 0.110

But 390 out of 540 clips received predicted probabilities below 0.13.

That is approximately 72% of the dataset.

The model concentrated most of its predictions in a narrow low-probability region.

Why is this important?

Imagine a model that predicts every clip has a 15% probability of containing an incident.

If the actual incident rate is also 15%, its aggregate calibration could be perfect.

But it wouldn't distinguish which clips actually contain incidents.

It would simply predict the overall prevalence.

This illustrates a limitation of calibration metrics: they measure agreement between probabilities and observed frequencies, not necessarily whether confidence is useful for individual predictions.

The video model's low ECE is therefore encouraging, but we need further analysis before claiming that it provides a useful confidence signal.

Why does this lesson matter?

In a real urban monitoring system, we don't just want a model that produces probabilities matching the average incident rate.

We want to know:

- Which predictions are likely to be correct?
- Which predictions are likely to be wrong?
- Can the system identify uncertain cases?
- Can it send uncertain cases to human reviewers?
- Can we safely automate decisions for high-confidence cases?

A single aggregate calibration score cannot answer all these questions.

That's why W10.3 is important.

## The connection between all four lessons

## What Week 10 taught us

1. Architecture doesn't guarantee reliability.

   More complex models can learn useful patterns while producing less trustworthy confidence estimates.
2. Performance doesn't automatically transfer.

   A model calibrated on one dataset can become poorly calibrated when the environment changes.
3. Prediction quality has multiple dimensions.

   Ranking examples correctly and estimating their probabilities accurately are different capabilities.
4. Aggregate metrics don't tell the whole story.

   A model can have a good ECE while providing little useful information about individual predictions.

The bigger engineering lesson:

Building a good ML model isn't just about maximizing accuracy or ROC-AUC. It's about understanding when its predictions are reliable, when its confidence is meaningful, how it behaves under distribution shift, and when its outputs should or should not be trusted by downstream systems.


---
---
---
---
---
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 2
UrbanMind-AI