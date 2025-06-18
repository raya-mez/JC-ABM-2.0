"""Library for Simulated Annealing for Optimality Theory"""

import math
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ----- Utility functions for candidate formatting -----
def flatten_singletons(x):
    """Recursively flatten singleton lists, including nested ones.
    Examples:
        [['V']]           -> 'V'
        ['SN', ['V']]     -> ['SN', 'V']
        [[['SN']], [['V']]] -> ['SN', 'V']
    """
    # Flatten the top level if it's a singleton list
    while isinstance(x, list) and len(x) == 1:
        x = x[0]

    # If it's still a list, apply recursively to each element
    if isinstance(x, list):
        return [flatten_singletons(elem) for elem in x]
    else:
        return x


def flatten_sf(sf):
    """Utility to flatten nested list trees using recursion.
    
    This is because candidate surface forms with the same linear structure
    are assigned the same number of violation marks by all constraints. 
    
    Args:
        sf (list or str): The surface form, which can be a nested list or the 'V' string.

    Returns:
        list: A flat list of surface form string elements.
    """
    if isinstance(sf, str):
        return [sf]
    elif isinstance(sf, list):
        return [item for sublist in sf for item in flatten_sf(sublist)]
    else:
        return []


def prettify_sf(sf, flat=False):
    """Recursively convert nested lists to string with bracketed nested or linear structure."""
    if flat:
        sf = flatten_sf(sf)
    if isinstance(sf, (tuple, list)):
        return "[" + " ".join(prettify_sf(sub) for sub in sf) + "]"
    return str(sf)
# --------------------------------- 

# ----- Neighborhood generation -----
def generate_neighbors(sf):
    """Generate neighborhood sf's of a candidate. 

    Args:
        sf (list or str): Surface form as nested list (or 'V' string).

    Returns:
        list of lists (or str): Neighbor candidates as nested lists (or 'V' string).
    """
    # Validate input
    assert isinstance(sf, (list, str)), "Input must be a list or the string 'V'."
    if isinstance(sf, str) and sf != 'V':
        raise ValueError("Input must be 'V' or a (nested) list.")    
    
    # Process input surface form
    sf = flatten_singletons(sf)

    neighbors = []

    # Basic step 1: Add 'SN' at beginning
    neighbors.append(['SN', sf])
    # Basic step 2: Add 'SN' at end
    neighbors.append([sf, 'SN'])

    # Basic step 3: Remove outermost SN        
    if sf[0] == 'SN':
        n = sf[1:]
        n = flatten_singletons(n) if len(n) == 1 else n 
        neighbors.append(n)
    elif sf[-1] == 'SN':
        n = sf[:-1]
        n = flatten_singletons(n) if len(n) == 1 else n
        neighbors.append(n)

    # Basic step 4a: Reverse any subtree
    for i, subtree in enumerate(sf):
        if not isinstance(subtree, str) and len(subtree) > 1:
            reversed_sub = subtree[::-1]
            new_sf = sf[:i] + [reversed_sub] + sf[i+1:]
            neighbors.append(new_sf)
    
    # Basic step 4b: Reverse top-level daughters if more than 1
    if not isinstance(sf, str) and len(sf) > 1:
        neighbors.append(sf[::-1])
    
    return neighbors
# --------------------------------- 

# ----- Constraint functions -----
def faith_neg(sf):
    """Faith[Neg]: (Faithfulness constraint) Penalize if negation is not expressed in surface form"""
    return 0 if 'SN' in flatten_sf(sf) else 1

def star_neg(sf):
    """*Neg: (Markedness constraint) Penalize for every negation marker"""
    return flatten_sf(sf).count('SN')

def neg_first(sf):
    """NegFirst: (Markedness constraint) Penalize if no negation at beginning"""
    return 0 if flatten_sf(sf)[0] == 'SN' else 1

def neg_last(sf):
    """NegLast: (Markedness constraint) Penalize if no negation at end"""
    return 0 if flatten_sf(sf)[-1] == 'SN' else 1
# --------------------------------- 

# ----- Constraint set evaluation -----
CONSTRAINTS_LIST = [faith_neg, star_neg, neg_first, neg_last]

def eval_constraints(sf, constraints=CONSTRAINTS_LIST):
    """Evaluate a given surface form against all constraints in the specified list.

    Args:
        sf (list or str): Surface form as nested list (or 'V' string).
        constraints (list): List of constraint functions to apply.

    Returns:
        list: List of violation counts for each constraint in order.
    """
    return [c(sf) for c in constraints]
# --------------------------------

# ----- Hierarchies and K-values -----
HIERARCHIES_DICT = {
    "H1": [faith_neg, star_neg, neg_first, neg_last],  # preverbal pure
    "H2": [faith_neg, neg_first, star_neg, neg_last],  # preverbal mixed
    "H3": [faith_neg, neg_first, neg_last, star_neg],  # discontinuous (left)
    "H4": [faith_neg, neg_last, neg_first, star_neg],  # discontinuous (right)
    "H5": [faith_neg, neg_last, star_neg, neg_first],  # postverbal mixed
    "H6": [faith_neg, star_neg, neg_last, neg_first]   # postverbal pure
}

K_VALUES_DEFAULT = [4, 3, 2, 1]
# ---------------------------------


# ----- Simulated Annealing Algorithm -----
def sa_ot(initial_sf, grammar, 
          K_max=5, K_step=1, t_max=3, t_min=0, t_step=1, max_no_moves=50,
          verbose=False):
    """Run simulated annealing with optimality theory.

    Args:
        initial_sf (str): The initial surface form.
        grammar (dict): A dictionary mapping constraints to their K-values.
        K_max (int, optional): The maximum K-value. Defaults to 5.
        K_step (int, optional): The step size for K. Defaults to 1.
        t_max (int, optional): The maximum temperature. Defaults to 3.
        t_min (int, optional): The minimum temperature. Defaults to 0.
        t_step (int, optional): The step size for temperature. Defaults to 1.
        max_no_moves (int, optional): The maximum number of moves without improvement. Defaults to 50.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        tuple: The final surface form after simulated annealing.
    """
    current = initial_sf

    # Iterate over temperature schedule
    K = K_max
    no_moves_count = 0
    iters = 0
    
    # Temperature K will decrease until max_no_moves is reached
    while no_moves_count < max_no_moves:
        # Iterate over t temperatures
        for t in np.arange(t_max, t_min, -t_step):
            iters += 1
            if verbose:
                print(f"\n>>> Iteration {iters}, K={K}, t={t}, Current: {current}")
            
            # Randomly select a neighbor of the current candidate
            cand = random.choice(generate_neighbors(current))
            if verbose:
                print(f"Candidate: {cand}")
            
            # Find the fatal constraint
            sorted_grammar = sorted(grammar.items(), key=lambda x: x[1], reverse=True)
            sorted_constraints = [constraint for constraint, _ in sorted_grammar]
        
            for constraint in sorted_constraints:
                if verbose: 
                    print(f"Evaluating constraint: {constraint.__name__}")
                c_curr, c_cand = constraint(current), constraint(cand)
                if verbose:
                    print(f"Current: {c_curr}, Candidate: {c_cand}")
                if c_curr != c_cand:
                    no_fatal = False # Reset no_fatal flag
                    C = constraint
                    # Get the K-value for the fatal constraint
                    k_C = grammar[C]
                    # Calculate the difference in violations for the fatal constraint between current and candidate
                    d = c_cand - c_curr 
                    if verbose:
                        print(f"Fatal constraint: {C.__name__}, K-value: {k_C}, Difference: {d}")
                    break
                else:
                    no_fatal = True  # No fatal constraint found
            
            # Decide whether to move to the candidate
            e = 1e-9 if t == 0 else 0  # to avoid division by zero in probability calculation
            
            if d < 0 or no_fatal:  # if candidate is more harmonic or no fatal constraint (equal harmony)
                current = cand # move to not-less harmonic neighbor
                no_moves_count = 0 # reset no_moves if a move is made
                if verbose:
                    print("Moved to not-less harmonic neighbor.")
            
            else:
                if k_C < K:
                    current = cand # move to less harmonic neighbor if k-value of fatal constraint is lower than temperature K
                    no_moves_count = 0 # reset no_moves if a move is made
                    if verbose:
                        print(f"Moved to less harmonic neighbor because k_C={k_C} < K={K}.")
                elif k_C == K:
                    p = random.random()
                    if p < math.exp(-d / (t + e)):
                        current = cand # move to less harmonic neighbor with probability exp(-d / (t + e))
                        no_moves_count = 0 # reset no_moves if a move is made
                        if verbose:
                            print("Moved to less harmonic neighbor with because p={p} < exp(-d / (t + e)).")
                else:
                    no_moves_count += 1  # increase no_moves if no move is made
                    if verbose:
                        print("Did not move")
        
        K -= K_step  # Decrease K 
    
    if verbose:
        print(f"\nFinal candidate found after {iters} iterations.")
    
    if star_neg(current) > 2:
        print(f"Warning: Final candidate {current} has {star_neg(current)} *Neg violations (should be <= 2).")
    
    return current
# ---------------------------------

