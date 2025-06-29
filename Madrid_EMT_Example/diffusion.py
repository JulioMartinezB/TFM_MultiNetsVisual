from pymnet import draw
from Multinets.diffusion import SIR_net_diffusion
import matplotlib.pyplot as plt
from Madrid_EMT_Example.utils import read_multiplex_edges_from_csv,EDGES_PATH,MADRID_NODE_COORDS





def simulate_and_plot_SIR_diffusion(
    mnet,
    node_coords,
    initial_infected_nodes,
    output_dir='imgs/SIR/',
    iterations=10,
    interlayers_beta=None,
    intra_gamma=None,
    default_node_size=0.01
):
    """
    Simulates SIR diffusion on a multilayer network and plots each time step.

    Args:
        mnet (MultilayerNetwork): The weighted multilayer network.
        node_coords (dict): Node coordinate dictionary.
        initial_infected_nodes (list of tuple): List of (node, layer) tuples to infect at time=0.
        output_dir (str): Directory to save output plots.
        iterations (int): Number of diffusion steps to simulate.
        interlayers_beta (dict, optional): Infection rates between layers.
        intra_gamma (dict, optional): Recovery rates within each layer.
        default_node_size (float): Uniform node size in visualization.
    """

    # Default beta/gamma if not provided
    if interlayers_beta is None:
        interlayers_beta = {
            (('metro',), ('metro',)): 65000,
            (('metro',), ('urban',)): 4,
            (('metro',), ('interurban',)): 0.9,
            (('urban',), ('urban',)): 50000,
            (('urban',), ('interurban',)): 10000000,
            (('urban',), ('metro',)): 4,
            (('interurban',), ('interurban',)): 50000,
            (('interurban',), ('metro',)): 4,
            (('interurban',), ('urban',)): 10000000
        }

    if intra_gamma is None:
        intra_gamma = {
            ('metro',): 0.1,
            ('urban',): 0.2,
            ('interurban',): 0.2
        }

    # Initialize node state: 0 = susceptible, 1 = infected, 2 = recovered
    initial_state = {
        node_layer: 0 for node_layer in mnet.iter_node_layers()
    }
    for node_layer in initial_infected_nodes:
        initial_state[node_layer] = 1

    # Run diffusion simulation
    final_states = SIR_net_diffusion(
        mnet,
        interlayers_beta=interlayers_beta,
        intra_gamma=intra_gamma,
        iterations=iterations,
        initial_state=initial_state
    )

    # Setup node sizes
    nodeSizeDict = {
        node_layer: default_node_size for node_layer in mnet.iter_node_layers()
    }

    # Plot each iteration
    for i, state in enumerate(final_states):
        fig = plt.figure(figsize=(36, 24))
        ax = fig.add_subplot(111, projection='3d')

        nodeColorDict = {
            node_layer: 'red' if state[node_layer] == 1 else
                        'blue' if state[node_layer] == 2 else
                        'gray'
            for node_layer in mnet.iter_node_layers()
        }

        nodeLabelDict = {
            node_layer: node_layer[0] if initial_state[node_layer] == 1 else ''
            for node_layer in mnet.iter_node_layers()
        }

        draw(
            mnet,
            nodeCoords=node_coords,
            nodeColorDict=nodeColorDict,
            nodeLabelDict=nodeLabelDict,
            nodeLabelRule={},
            defaultNodeLabel="",
            defaultEdgeAlpha=0.4,
            nodeSizeDict=nodeSizeDict,
            layergap=1,
            ax=ax
        )

        plt.title(f"State after {i} iterations")
        plt.savefig(f'{output_dir}/sir_state_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

mnet=read_multiplex_edges_from_csv(EDGES_PATH,weighted=True),

interlayers_beta = {
    (('metro',),('metro',)): 65000,
    (('metro',), ('urban',)): 4,
    (('metro',), ('interurban',)): 0.9,

    (('urban',), ('urban',)): 50000,
    (('urban',), ('interurban',)): 10000000,
    (('urban',), ('metro',)):4,

    (('interurban',), ('interurban',)): 50000,
    (('interurban',), ('metro',)): 4,

    (('interurban',), ('urban',)): 10000000
}
intra_gamma = {
    ('metro',): 0.1,
    ('urban',): 0.2,
    ('interurban',): 0.2
}



simulate_and_plot_SIR_diffusion(
    mnet,
    MADRID_NODE_COORDS,
    initial_infected_nodes=[('AVENIDA DE AMERICA', 'metro'),('PLAZA ELIPTICA', 'urban')],
    output_dir='imgs/SIR',
    iterations=10,
    interlayers_beta=interlayers_beta,
    intra_gamma=intra_gamma,
    default_node_size=0.01
)
