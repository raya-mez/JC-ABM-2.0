"""Library for Simulated Annealing for Optimality Theory"""

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----- Example Surface Forms (SF) -----
SF_EXAMPLES = [
    'V',
    ['V'],
    ['SN', 'V'],
    [['V'], 'SN'],
    [['SN', [['V', 'SN']]]],
    [['V', 'SN'], 'SN'],
    ['SN', ['SN', 'V']],
    [[['SN', 'V'], 'SN'], 'SN'],
]

# ----- Utility functions for candidate formatting -----
def flatten_singletons(x):
    """Recursively flatten singleton lists, including nested ones.
    Examples:
        [['V']]           -> 'V'
        ['SN', ['V']]     -> ['SN', 'V']
        [[['SN']], [['V']]] -> ['SN', 'V']
    
    Used in neighborhood generation where we want to 
    remove unnecessary bracketing around singleton lists,
    without flattening the whole structure.  
    """
    # Flatten the top level if it's a singleton list
    while isinstance(x, list) and len(x) == 1:
        x = x[0]

    # If it's still a list, apply recursively to each element
    if isinstance(x, list):
        return [flatten_singletons(elem) for elem in x]
    
    # If it's not a list, return as is
    else:
        return x


def flatten_sf(sf):
    """Utility to flatten nested list trees using recursion.
    
    Args:
        sf (list or str): The surface form, which can be a nested list or the 'V' string.

    Returns:
        list: A flat list of surface form string elements.
    
    
    Used in constraint evaluation as candidate surface forms with the same linear structure
    are assigned the same number of violation marks by all constraints. 
    """
    if isinstance(sf, str):
        return [sf]
    elif isinstance(sf, list):
        return [item for sublist in sf for item in flatten_sf(sublist)]
    else:
        return []


def serialize_sf(sf, flat=False):
    """Recursively convert nested lists to string with bracketed nested or linear structure."""
    sf = flatten_singletons(sf)  # Flatten singletons first
    if flat:
        sf = flatten_sf(sf)
    if isinstance(sf, (tuple, list)):
        return "[" + " ".join(serialize_sf(sub) for sub in sf) + "]"
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
K_VALUES_RANDOM = sorted([4.9] + [random.uniform(-0.1, 4.9) for _ in range(3)], reverse=True)
# ---------------------------------


# ----- Simulated Annealing Algorithm -----
def sa_ot(initial_sf, grammar, 
          K_max=5, K_step=1, t_max=3, t_min=0, t_step=1, max_no_moves=50,
          seed=None, verbose=False):
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
    if seed is not None:
        random.seed(seed)
    
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
                    print(f"    Current: {c_curr}, Candidate: {c_cand}")
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
                            print(f"Moved to less harmonic neighbor with because p={p} < exp(-d / (t + e)).")
                else:
                    no_moves_count += 1  # increase no_moves if no move is made
                    if verbose:
                        print("Did not move")
        
        K -= K_step  # Decrease K 
    
    if verbose:
        print(f"\nFinal candidate found after {iters} iterations. No-move count: {no_moves_count}. Final K: {K}")
    
    if star_neg(current) > 2:
        print(f"Warning: Final candidate {current} has {star_neg(current)} *Neg violations (should be <= 2).")
    
    return current
# ---------------------------------

def gla_update(grammar, predicted_sf, observed_sf, plasticity=0.1, update_ceiling=True):
    """
    Performs a single GLA update step.
    The goal is to update the K-values of the constraints in the grammar 
    such that its prediction better matches a target surface form. 

    Args:
        grammar (dict): Mapping of constraints to their ranking values {constraint_function: float}
        predicted_sf (list): predicted surface form (nested or not)
        observed_sf (list): observed surface form (nested or not)
        plasticity (float): learning rate
        update_ceiling (bool): whether to ensure ranks do not exceed a ceiling value of the highest k-value

    Returns:
        dict: updated ranks (same structure as input)
    """
    if predicted_sf == observed_sf:
        return grammar  # No update needed if predicted and observed are the same
    
    updated_grammar = grammar.copy()
    
    if update_ceiling:
        # Ensure k-values do not exceed a ceiling value of the highest k-value in the initial grammar
        max_k = max(grammar.values()) 
    
    for constraint_fn in grammar:
        # Calculate the difference in violation scores for the constraint
        predicted_violations = constraint_fn(predicted_sf)
        observed_violations = constraint_fn(observed_sf)
        diff = observed_violations - predicted_violations

        # If both violation scores are equal, no update is needed
        if diff == 0:
            continue
        # If constraint favors prediction, decrease its k-value by one step (target has lower violation score than prediction)
        if diff > 0:
            updated_grammar[constraint_fn] -= plasticity
        # If constraint favors target, increase its k-value by one step 
        else:
            updated_grammar[constraint_fn] += plasticity 
            # If applicable, ensure the new k-value does not exceed the ceiling value
            if update_ceiling and updated_grammar[constraint_fn] >= max_k:
                excessive_k = updated_grammar[constraint_fn]
                updated_grammar[constraint_fn] = max_k - 1e-9  # Set to just below the ceiling value
                print(f"WARNING: Rank of {constraint_fn.__name__} exceeded ceiling value of {max_k} ({excessive_k}) -> set to {max_k - 1e-9}.")

    return updated_grammar


# ----- Utility functions (GLA) -----
def grammar_dict_to_readable(grammar_dict):
    """Convert dict with function keys to dict with function name keys for display."""
    return {fn.__name__: k for fn, k in grammar_dict.items()}


def get_hierarchy_name(grammar):
    """Get the name of the current hierarchy based on the grammar."""
    sorted_grammar = sorted(grammar.items(), key=lambda x: x[1], reverse=True)
    sorted_constraints = [constraint for constraint, _ in sorted_grammar]
    
    for h_name, constraints in HIERARCHIES_DICT.items():
        if constraints == sorted_constraints:
            return h_name
    return "Unknown hierarchy"
# ---------------------------------


# ----- Agent class -----
class SAOTAgent:
    def __init__(self, id, hierarchy_name, k_values):
        """An agent equipped with:
        - an OT grammar (the model of its competence): A set of constraints with associated K-values. 
                                                        The agent's grammar is initialized with random K-values between -0.1 and 4.9 
                                                        for the constraints in the order of the specified hierarchy 
                                                        (only the Faith[Neg] constraint is always initialized with a K-value of 4.9).
        - an SA-OT production procedure (performance): The agent can produce surface forms based on its grammar.
        - a GLA learning procedure (Boersma, 1997): The agent can update its grammar based on observed surface forms. 
        
        Args:
            id (int): Unique identifier for the agent.
            hierarchy_name (str): Name of the hierarchy to initialize the agent's grammar.
            k_values (list or str): List of 4 K-values for the constraints in the hierarchy, or string choosing a predefined list. 
                                    If 'random', uses random values between -0.1 and 4.9 (except for Faith[Neg] which is always initialized with 4.9).
                                    If 'default', uses the default K-values [4, 3, 2, 1].
        """
        
        self.id = id

        self.HIERARCHIES_DICT = {
            "H1": [faith_neg, star_neg, neg_first, neg_last],
            "H2": [faith_neg, neg_first, star_neg, neg_last],
            "H3": [faith_neg, neg_first, neg_last, star_neg],
            "H4": [faith_neg, neg_last, neg_first, star_neg],
            "H5": [faith_neg, neg_last, star_neg, neg_first],
            "H6": [faith_neg, star_neg, neg_last, neg_first]
        }
        
        self.global_optima = {
            "H1": ['SN', 'V'],
            "H2": ['SN', 'V'],
            "H3": ['SN', 'V', 'SN'],
            "H4": ['SN', 'V', 'SN'],
            "H5": ['V', 'SN'],
            "H6": ['V', 'SN']
        }
        
        self.local_optima_produced = {h: [] for h in self.HIERARCHIES_DICT.keys()}

        # Initialize the k-values of the constraints
        if k_values == 'random':
            # Initialize K-values of markedness constraints randomly, only faithfulness constraint Faith[Neg] always has K=4.9
            self.k_values = sorted([4.9] + [random.uniform(-0.1, 4.9) for _ in range(3)], reverse=True)
        elif k_values == 'default':
            # Use the default K-values [4, 3, 2, 1]
            self.k_values = K_VALUES_DEFAULT
        else:
            # Use custom K-values
            assert isinstance(k_values, list) and len(k_values) == 4, "k_values must be a list of 4 K-values."
            self.k_values = k_values
        
        # Record the chosen hierarchy
        self.initial_hierarchy_name = hierarchy_name 
        
        # Initialize agent's grammar with the specified hierarchy and k-values scheme
        self.grammar = self.init_grammar()

    def __repr__(self):
        """String representation of the agent."""
        return f"Agent(id={self.id}, hierarchy={self.hierarchy_name}, grammar={self.grammar_dict_to_readable(self.grammar)})"

    def init_grammar(self):
        """Initialize the agent's grammar based on the specified hierarchy."""
        assert self.initial_hierarchy_name in self.HIERARCHIES_DICT, f"Invalid hierarchy name: {self.initial_hierarchy_name}"
        
        init_k_values = self.k_values.copy()
        grammar = {constraint: k for constraint, k in 
                        zip(self.HIERARCHIES_DICT[self.initial_hierarchy_name], init_k_values)}
        return grammar 

    def grammar_dict_to_readable(self, grammar_dict):
        """Convert dict with function keys to dict with function name keys for display."""
        return {fn.__name__: rank for fn, rank in grammar_dict.items()}

    @property
    def hierarchy_name(self):
        """Dynamic property that returns current hierarchy name."""
        sorted_grammar = sorted(self.grammar.items(), key=lambda x: x[1], reverse=True)
        sorted_constraints = [constraint for constraint, _ in sorted_grammar]
        
        for h_name, h_constraints in self.HIERARCHIES_DICT.items():
            if h_constraints == sorted_constraints:
                return h_name
        return f"Unknown hierarchy: {self.grammar_dict_to_readable(self.grammar)}"
    
    def produce_sf(self):
        """Produce a surface form based on the agent's grammar."""        
        # Run SA-OT with current grammar state
        return sa_ot(initial_sf='V',
                     grammar=self.grammar,
                     K_max=5, K_step=1, t_max=3, t_min=0, t_step=1, max_no_moves=50)
    
    def learn_from(self, observed_sf, plasticity=0.1, verbose=False):
        """Update the agent's grammar based on observed surface form."""
        # Predict surface form using SA-OT with current grammar state
        predicted_sf = self.produce_sf()
        if verbose:
            print(f"Agent {self.id} learning based on predicted surface form: {serialize_sf(predicted_sf, flat=True)}")
            
        # Check if the predicted surface form matches the global optimum for the current hierarchy
        if serialize_sf(predicted_sf, flat=True) != serialize_sf(self.global_optima[self.hierarchy_name], flat=True):
            self.local_optima_produced[self.hierarchy_name].append(serialize_sf(predicted_sf, flat=True))
        
        # Update grammar based on comparison of predicted and observed sf
        updated_grammar = gla_update(grammar=self.grammar, predicted_sf=predicted_sf, observed_sf=observed_sf, plasticity=plasticity)
        self.grammar = updated_grammar


# ----- Model class -----
class SAOTModel:
    def __init__(self, pop_size=5, generations=100, 
                 gen0_hierarchy_name='H1', gen0_k_values='default',
                 k_values='random', gla_plasticity=0.1, seed=42,
                 learning_data_size=30, productions_per_agent=100):

        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)  
        
        self.pop_size = pop_size
        self.generations = generations
        self.gen0_hierarchy_name = gen0_hierarchy_name
        self.gen0_k_values = gen0_k_values
        self.k_values = k_values
        self.gla_plasticity = gla_plasticity  # Learning rate for GLA updates
        self.learning_data_size = learning_data_size
        self.productions_per_agent = productions_per_agent

        self.agents = []
        self.current_gen = 0
        
        self.production_history = {}
        self.initial_hierarchies = {}
        self.updated_hierarchies = {} 
        
        self.HIERARCHY_NAMES = [f"H{i}" for i in range(1, 7)]

    def __repr__(self):
        """String representation of the model."""
        return f"Model(seed={self.seed}, pop_size={self.pop_size}, generations={self.generations}, gen0_hierarchy={self.gen0_hierarchy_name}, k_values={self.k_values}, learning_data={self.learning_data_size}"

    def init_pop(self):
        """Initialize a population of agents with specified initial hierarchy."""  
        for i in range(-self.pop_size, 0):
            agent = SAOTAgent(i, hierarchy_name=self.gen0_hierarchy_name, k_values=self.gen0_k_values) 
            self.agents.append(agent)
    
    def step(self):
        """Perform a single step of the model (i.e., one generation):
        - create new generation of agents with randomly initialized grammars, 
        - adjust their grammars iteratively based on learning data from the previous generation, 
        - record a production sample from the new generation once learning is done,
        - replace the old generation with the new one.
        """       
        assert len(self.agents) > 0, "Population must be initialized before stepping."
        self.current_gen += 1
        
        # Initialize new agents with a randomly chosen hierarchy
        new_agents = []
        id_start = (self.current_gen-1) * len(self.agents) + 1
        for id in range(id_start, id_start + self.pop_size):
            hierarchy_name = random.choice(self.HIERARCHY_NAMES) # Randomly select a hierarchy for the new agent
            agent = SAOTAgent(id, hierarchy_name=hierarchy_name, k_values=self.k_values) 
            new_agents.append(agent)
        
        # Store initial hierarchies for this generation
        self.initial_hierarchies[self.current_gen] = [agent.hierarchy_name for agent in new_agents]
        
        # Each agent learns from (updates their grammar given) observed surface forms produced by the previous generation
        for agent in new_agents: 
            # Each agent is exposed to learning_data_size productions of the previous generation
            for _ in range(self.learning_data_size):
                # Randomly select an adult to learn from
                adult = random.choice(self.agents)
                adult_sf = adult.produce_sf()  # Adult produces a surface form
                agent.learn_from(adult_sf, self.gla_plasticity)  # Update agent's grammar based on the adult's production
        # After learning, agents in the new generation will produce surface forms based on their updated grammars

        # Store final hierarchies for this generation
        self.updated_hierarchies[self.current_gen] = [agent.hierarchy_name for agent in new_agents]

        # Record production sample from new generation with updated grammars
        production_sample = []
        for agent in new_agents:
            # Each agent produces a sample of surface forms based on its grammar
            for i in range(self.productions_per_agent):
                production = agent.produce_sf()
                production_sample.append(production)  
        assert len(production_sample) == 500, f"Production sample size is {len(production_sample)}, expected 500."
        self.production_history[self.current_gen] = production_sample

        # Replace old generation of agents with the new one
        self.agents = new_agents

    def run(self, generations=None):
        """Run the model for a specified number of generations."""
        # Initialize the preliminary generation of agents (index 0)
        self.init_pop()
        print(f"Model initialized with {self.pop_size} agents.")
        
        # Run the model for the specified number of generations
        generations = generations if generations is not None else self.generations
        for i in range(generations):
            if i % 10 == 0:
                print(f"Running generation {i+1}/{generations}...")
            self.step()
        print(f"Model run completed after {self.generations} generations.")
    

    def plot_history(self, pretty=True, flat=True):
        """Plot the evolution of productions over generations."""
        if pretty:
            # Flatten the sf's in the production history to a list of surface forms
            history = {gen: [serialize_sf(sf, flat=flat) for sf in productions]
                        for gen, productions in self.production_history.items()}
        else:
            history = {gen: [str(sf) for sf in productions]
                        for gen, productions in self.production_history.items()}
        
        # Convert production history dict to DataFrame
        history_df = pd.DataFrame(history)  # rows: surface forms, columns: generations
        # Count the occurrences of each surface form per generation 
        history_count = history_df.apply(lambda x: x.value_counts()).fillna(0)
        # Convert counts to proportions
        history_proportions = history_count.apply(lambda x: x / x.sum(), axis=0) 
        # Display the proportions of surface forms produced by agents in each generation
        display(history_proportions)

        # Plotting the results over generations 
        plt.figure(figsize=(12, 6))
        generations = history_proportions.columns 
        for sf in history_proportions.index: 
            proportions = history_proportions.loc[sf]  # Get proportions for this SF across generations
            plt.plot(generations, proportions, marker='o', label=sf, linewidth=2)

        plt.title("Evolution of Surface Forms Over Generations")
        plt.xlabel("Generation")
        plt.xticks(range(0, len(generations)+1, 10))
        plt.ylabel("Proportion")
        plt.ylim(0, 1)  # Set y-axis limits from 0 to 1 for proportions
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()