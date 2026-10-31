
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
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 2
UrbanMind-AI