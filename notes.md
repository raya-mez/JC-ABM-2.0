# Bayesian Model

## Extention
- preverbal bias in priors (typological tendency + starting point of the cycle)

## Notes: Minimal Bayesian Model (vs. Bayesian Variant)
The models are mathematically equivalent but computationally different due to the Mesa Agent-Based Modeling framework in the original model which creates different stochastic dynamics:
- Different random number consumption patterns
- Different agent scheduling/activation orders
- Different memory allocation patterns affecting floating-point precision
The transitions are an emergent property of the specific computational framework, not just the underlying mathematical model.


---------------------------------------------------------------------------------------
# SA-OT 

## To do: Extension
- [ ] Test if transition 2 occurs when setting initial grammar to discontinuous
- [ ] Play with SA parameters (temeprature) -> try to obtain cyclicity

## To do: Checks & debugging
- [ ] Check surprising results of convergence test
    - H4-6 have a very consistent convergence time over runs (very small SD). 
        - Especially H4 and H6 when using default k-values: then SD=0 (see saved plots). 
        - SD(H4) > SD(H5) > SD(H6) when using random k-values (see saved plots).
    - H2 (high convergence time) vs. H6 (low convergence time): Might have sth to do with whether the hierarchy represents a pure or a mixed stage -> if mixed (e.g., H2), might produce erroneous forms that are more distant and require more update iterations, vs. if pure (e.g., H6) they always produce global optimum. 
    - H2 (slow convergence) vs. H5 (fast convergence): both are mixed, but H2 is closer to H1 than H5 is. So how come?
        - Exactly because H2 is closer but mixed stage ith global optimum same as H1 -> will often produce [SN V] and thus skip learning update, while H5 produces [V SN] or [SN V SN], which are both different from target, thus learning happens at every step, speeding up convergence. 
    - H3 (very high convergence time) vs. H4 (low convergence time), even though H4 is the most distant from H1 (in both directions).
        - sometimes H4 does not transition through other (mixed) hierarchies, but directly arrives at H1 by GLA updates
        - or transition can go through H5 and H6, which are faster to converge because have higher error rate giving constant teaching signal (as explained above) vs H3 gets stuck in H2.
- [x] Repeat convergence test for random float K-values between -0.1 and 4.9
- [x] Retest original January model (no modifications)
    - Does not always produce the expected evolution on individual runs (e.g., single run on seed=42 transitions: postverbal -> preverbal -> discontinuous (stable))
    - But the mean over 20 simulations has the expected trend (transition from preverbal to discontinuous, as in my report) 
- [ ] Check modifications of original January model one-by-one
    - [x] K-values (random floats between -0.1 and 4.9)
        - Much more fluctuating results (less steady evolution) over single run and mean over 20 runs
        - Does not stabilize at discontinuous
    - [ ] different starting hierarchies
    - [ ] convergence test
- [ ] 

## Notes
- In January version of the model, I had missed footnote 4 on page 14 that specifies the initialization of the grammar of new agents: 
> Constraint Faith\[Neg] was assigned rank 4.9, and the markedness constraints were associated with a random floating point value between -0.1 and 4.9. The standard parameters of the SA-OT Algorithm (K_max = 5, K_step = 1, t_step = 1, etc.) were used, as discussed in the Appendix."
- Unclear initialization of K-values for gen 0. 
- 1.0 and 2.0 versions of the sa_ot function run the same way given same seed.
- Flaw of the SA-OT model: pure discontinuous stages is repr by 2 hierarchies vs the other 2 pure stages are repr by only 1 hierarchy each.
- 