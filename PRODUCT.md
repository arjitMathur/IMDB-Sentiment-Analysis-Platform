# Product

## Register

product

## Platform

web

## Users

Primary: portfolio and demo visitors who want to paste a movie review, see a clear positive/negative call, and compare how models disagree. Secondary: ML engineers and students who open the same surface to check latency, confidence, and benchmark metrics while testing models.

## Product Purpose

Replace the Streamlit demo with a Next.js interface that talks to the existing FastAPI backend. Users analyze review sentiment across baseline, LSTM, and DistilBERT models, then inspect confidence, latency, and stored benchmarks. Success is a fast, trustworthy analyze flow that still exposes enough model detail for technical visitors.

## Positioning

Paste a review. See how three models actually disagree.

## Brand Personality

Precise, cinematic, confident. Voice is plain and technical when needed, never hype. The interface should feel like a dark screening room with a sharp readout, not a marketing pitch.

## Anti-references

Purple AI-dashboard glow, Streamlit-default chrome, generic SaaS gradient cards, hero-metric templates, and decorative glassmorphism.

## Design Principles

1. **Show the verdict first.** Label, confidence, and model identity land before secondary metrics.
2. **Disagreement is the product.** Side-by-side model comparison is a first-class action, not a footnote.
3. **Demo-first, engineer-ready.** The analyze path is the hero; metrics and latency stay available without cluttering the first action.
4. **Earn trust with state.** Loading, empty, error, and untrained-model states are explicit and recoverable.
5. **Restraint over spectacle.** Premium dark surfaces, one accent, motion only for feedback.

## Accessibility & Inclusion

WCAG AA contrast for text and controls. Respect `prefers-reduced-motion`. Keyboard-reachable analyze and model controls. Color is never the only signal for positive vs negative (pair with label text).
