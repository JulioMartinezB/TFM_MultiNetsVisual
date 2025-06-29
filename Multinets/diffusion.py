import numpy as np

def update_state(nl1_state, nl2_state,gamma_nl1,gamma_nl2,beta_nl1_nl2,beta_nl2_nl1):
    """
    Updates the infection states of two nodes in a multilayer SIR model.

    This function uses a probabilistic rule based on the current states of
    two nodes, their recovery rates (`gamma`) and infection rates (`beta`)
    to simulate one step of SIR dynamics.

    Args:
        nl1_state (int): Current state of node 1 (0: susceptible, 1: infected, 2: recovered).
        nl2_state (int): Current state of node 2.
        gamma_nl1 (float): Recovery probability for node 1.
        gamma_nl2 (float): Recovery probability for node 2.
        beta_nl1_nl2 (float): Infection probability from node 1 to node 2.
        beta_nl2_nl1 (float): Infection probability from node 2 to node 1.

    Returns:
        tuple: The new states (nl1_state, nl2_state) after one interaction step.
    """
    match (nl1_state,nl2_state):

        case (1,1):
            return (np.random.choice([1,2], p=[1-gamma_nl1,gamma_nl1]),
                    np.random.choice([1,2], p=[1-gamma_nl2,gamma_nl2])) 
        case (0,1):
            return (np.random.choice([0,1], p=[1-beta_nl2_nl1,beta_nl2_nl1]),
                    np.random.choice([1,2], p=[1-gamma_nl2,gamma_nl2]))
        case (1,0):
            return (np.random.choice([1,2], p=[1-gamma_nl1,gamma_nl1]),
                    np.random.choice([0,1], p=[1-beta_nl1_nl2,beta_nl1_nl2]))
        
        case (1,2):
            return (np.random.choice([1,2], p=[1-gamma_nl1,gamma_nl1]),
                    2)
        case (2,1):
            return (2,
                    np.random.choice([1,2], p=[1-gamma_nl2,gamma_nl2]))
        case default:
            return (nl1_state,nl2_state)

def SIR_net_diffusion(self, interlayers_beta , intra_gamma, iterations, initial_state=None):
    """
    Simulates SIR diffusion dynamics on a multilayer network over a number of iterations.

    Each node in the multilayer network is identified by a (node, layer) tuple. The state
    of each node can be:
        - 0: Susceptible
        - 1: Infected
        - 2: Recovered

    The diffusion model incorporates both intra-layer and inter-layer infection and recovery
    dynamics. Infection is probabilistic and modulated by edge weights.

    Args:
        interlayers_beta (dict): A dictionary with keys ((layer1), (layer2)) representing layer tuples,
            and values representing infection probabilities between those layers.
        intra_gamma (dict): A dictionary with keys (layer,) representing individual layers,
            and values as the recovery probabilities (gamma) for those layers.
        iterations (int): Number of time steps to simulate.
        initial_state (dict, optional): A dictionary mapping (node, layer) tuples to initial states.
            If None, a random initial state is generated (80% susceptible, 20% infected).

    Returns:
        list of dict: A list where each element is the node state dictionary at a given iteration.

    Notes:
        - Diffusion stops early if no node changes state or all nodes are either susceptible or recovered.
        - The function uses edge weights to scale infection probabilities.
    """
    
    if not initial_state:
        initial_state = {edge_nodes: np.random.choice([0,1],p=[0.8,0.2]) for edge in self.edges for edge_nodes in self._link_to_nodes(edge[:-1])}

    state_list = [initial_state.copy()]
    state = initial_state

    for i in range(iterations):
        state_changed = False  # Track if any node state changes
        new_state = state.copy()
        for edge in self.edges:
            ((nl1_0,nl1_1, weight),nl2) = self._link_to_nodes(edge)
            nl1 = (nl1_0, nl1_1)
            nl1_state = state[nl1]
            nl2_state = state[nl2]

            gamma_nl1 = intra_gamma[tuple(nl1[1:])]
            gamma_nl2 = intra_gamma[tuple(nl2[1:])]
            beta_nl1_nl2 = interlayers_beta[(nl1[1:], nl2[1:])]
            beta_nl2_nl1 = interlayers_beta[(nl2[1:], nl1[1:])]


            # Scale beta by weight
            beta_nl1_nl2 = min(beta_nl1_nl2 * weight, 1.0)
            beta_nl2_nl1 = min(beta_nl2_nl1 * weight, 1.0)

            updated_nl1, updated_nl2  = update_state(
                nl1_state, nl2_state,
                gamma_nl1, gamma_nl2,
                beta_nl1_nl2, beta_nl2_nl1
            )

            if new_state[nl1] != updated_nl1:
                new_state[nl1] = updated_nl1
                state_changed = True

            if new_state[nl2] != updated_nl2:
                new_state[nl2] = updated_nl2
                state_changed = True

        # Stop if nothing changed
        if not state_changed:
            break


        state = new_state
        state_list.append(state.copy())
        #if all states are 0 or 2, then the epidemic is over
        if all(s in (0, 2) for s in state.values()):
            break
            
    return state_list
            
