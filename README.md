# Agent-Based Bayesian Model of Jespersen's Cycle


# Requirements
```bash
conda env create -f environment.yml
conda activate jc_abm_2.0
```

# References for Modeling Decisions
| Component                     | Source / Rationale                                                          |
| ----------------------------- | --------------------------------------------------------------------------- |
| Constraint-based grammars     | Optimality Theory (Prince & Smolensky 1993; de Swart 2009)                  |
| Iterated learning framework   | Kirby et al. (2007), Griffiths & Kalish (2005)                              |
| Bayesian learning             | Ferdinand & Zuidema (2008); Griffiths & Kalish (2005)                       |
| Constraint weights = K-values | Lopopolo & Biró (2011); this encodes hierarchy strength quantitatively     |
| Candidate space (\[NV], etc.) | de Swart (2009); simplified to 3 forms to match JC's diachronic pathway     |
| Likelihood via softmax        | Standard in probabilistic inference to convert utilities into probabilities |
| MAP strategy for agents       | More stable inference over generations (Ferdinand & Zuidema 2008)           |

