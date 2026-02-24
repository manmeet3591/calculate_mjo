# calculate_mjo

What this code does (standard RMM workflow)

Subset tropics (commonly 15°S–15°N) and average in latitude with cosine-latitude weighting

Remove climatology + first 3 harmonics of the annual cycle (typical WH method) to get anomalies

Standardize each variable by its own spatial/temporal std (training period)

Project the combined anomaly vector onto the two WH EOF patterns → gives RMM1/RMM2

Compute amplitude = sqrt(RMM1² + RMM2²) and phase (1–8)

Important: To get “official-like” RMM values, you need the same EOF patterns and normalization used by Wheeler & Hendon. You can either:

(A) compute EOFs yourself from a long training dataset (works, but won’t match “official” exactly), or

(B) use published WH EOF patterns / coefficients (recommended if you want to match).

The code below supports both: compute EOFs from training data, then apply to any target period.
