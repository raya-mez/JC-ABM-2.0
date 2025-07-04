# Simulated-Annealing for Optimality Theory (SAOT) Model (with Iterated Learning) of Jespersen's Cycle (JC)

# Overview
The goal of this project is to model Jespersen's cycle form - both at a synchronic and diachronic level. It replicates the Simulated Annealing for Optimality Theory (SAOT) model proposed by Lopopolo and Biró (2011) but implements it in Python instead of the OTKit software package (Biró 2010) they used. 
Thus, the project aimed to:
- test the replicability of the results of Lopopolo and Biro (2011) based on the description of their methods in the paper; 
- explore the different characteristics and parameters of the model;
- improve the model to account for the entire progression of JC, not only the first transition (from preverbal to discontinuous negation).

# Requirements
```bash
pip install numpy pandas matplotlib seaborn
```

# Project Structure
- `saot.py` contains variables and functions implementing model components
- `saot-demo.ipynb` is a Jupiter notebook demonstrating what these components represent and how the are implemented. It can be used as an illustrated introduction to SA-OT. 
- `exp1-jc_stages.ipynb` shows how SAOT models the different stages of JC.
- `exp3-jc_transitions.ipynb` contains an agent-based model using iterated learning to simulate the gradual transition from one pure stage of JC to the next.  
- `exp2-convergence_test.ipynb` shows the impact of the initial grammar of agents on their learning patterns.


# References
- Lopopolo, A., & Biro, T. (2011). Language Change and SA-OT: The case of sentential negation. Computational Linguistics in the Nether-
lands Journal. doi: 10.7282/T3BC3WKH.
- Biró, T. (2010), OTKit: Tools for Optimality Theory. A software package. http://www.birot.hu/OTKit/.
- Swart, H. de (2010), Expression and Interpretation of Negation: An OT Typology, Vol. 77 of Studies in Natural Language and Linguistic Theory, Springer, Dordrecht, etc., chapter 3: Markedness of Negation.


<!-- # References for Modeling Decisions
| Component                     | Source / Rationale                                                          |
| ----------------------------- | --------------------------------------------------------------------------- |
| Constraint-based grammars     | Optimality Theory (Prince & Smolensky 1993; de Swart 2009)                  |
| Iterated learning framework   | Kirby et al. (2007), Griffiths & Kalish (2005)                              |
| Constraint weights = K-values | Lopopolo & Biró (2011); this encodes hierarchy strength quantitatively      |
| Candidate space (\[NV], etc.) | de Swart (2009), Lopopolo & Biró (2011)                                     | -->
