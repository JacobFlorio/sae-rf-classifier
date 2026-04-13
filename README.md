# Sparse Autoencoders on an RF Signal Classifier

Apply sparse autoencoders to the hidden activations of a CNN trained to classify RF modulation schemes. Ask what "features" the network has learned to listen for — the EE-domain analogue of SAE work on language models.

## Research question
When an SAE is trained on the penultimate-layer activations of a modulation classifier, do the learned features correspond to interpretable signal properties (cyclostationarity periods, amplitude envelope statistics, phase transition rates), or do they just recover PCA directions in disguise? Does a sparser code improve or degrade adversarial robustness under jamming?

## Why this project
Interpretability work outside language is underexplored. RF is a domain with a strong classical-features baseline (cyclostationary analysis) — we can literally ask "did the SAE rediscover what a signal processing textbook would tell us to compute?" That's a rare, falsifiable interp question.

## Approach
1. Reuse the classifier from `neural-rf-frontend`.
2. Train SAEs at multiple sparsity levels on the residual stream.
3. For each learned feature: correlate with classical cyclostationary features.
4. Ablate features, measure per-class accuracy impact.
5. Jamming robustness: does sparsifying activations help?

## Deliverables
- `src/sae.py`
- `notebooks/feature_correlation.ipynb`
- `results/feature_atlas.md` — top-50 features annotated with classical DSP counterparts

## Sister project
`neural-rf-frontend` provides the base classifier.
