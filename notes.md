# Extensions
- preverbal bias in priors (typological tendency + starting point of the cycle)

---------------------------------------------------------------------------------------
# Minimal Bayesian Model (vs. Bayesian Variant)
The models are mathematically equivalent but computationally different due to the Mesa Agent-Based Modeling framework in the original model which creates different stochastic dynamics:
- Different random number consumption patterns
- Different agent scheduling/activation orders
- Different memory allocation patterns affecting floating-point precision
The transitions are an emergent property of the specific computational framework, not just the underlying mathematical model.


---------------------------------------------------------------------------------------
# SA-OT 
- I have missed footnote 4 on page 14 that specifies the initialization of the grammar of new agents: 
> Constraint Faith\[Neg] was assigned rank 4.9, and the markedness constraints were associated with a random floating point value between -0.1 and 4.9. The standard parameters of the SA-OT Algorithm (K_max = 5, K_step = 1, t_step = 1, etc.) were used, as discussed in the Appendix."

## To do: Improvements
- [ ] Test if transition 2 occurs when setting initial grammar to discontinuous
- [ ] Play with SA parameters (temeprature) -> try to obtain cyclicity

## Notes
- Unclear initialization of K-values for gen 0. 
- 1.0 and 2.0 versions of the sa_ot algorithm run the same way given same seed.