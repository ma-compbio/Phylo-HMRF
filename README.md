# Phylo-HMRF

Phylogenetic Hidden Markov Random Filed model

The command to use Phylp-HMRF for evolutionary state estimation is as follows. 

python phylo_hmrf.py [Options]

The options:

- -n, --num_states : the number of states to estimate for Phylo-HMRF, default = 10

- -p, --root_path : the directory where the input data files are kept, default = '.' (the current working directory)

- -f, --chromosome : the name of the chromosome to estimate hidden states using Phylo-HMRF, default = 1

- --num_neighbor : the number of neighbors of each node in the neighborhood system, default = 8

- -g, --estimate_type : the choice to consider edge weights in pairwise potential: 0: not consider edge weights; 3: consider edge weights, default = 3

- -b, --beta : the pairwise potentional parameter beta_0, default = 1

- --beta1 : the pairwise potential parameter beta_1, default = 0.1

- -r, --run_id : experiment id, default = 0

- -c, --cons_param : constraint parameter, default = 4

- -i, --initial_weight : initial weight for initial parameters, default = 0.5

- -j, --initial_magnitude : initial magnitude for initial parameters, default = 1

Example: python phylo_hmrf.py -f 1 -n 20 (using Phylo-HMRF to estimate 20 states on syntenic regions on chromosome 1)

For the provided example, the input includes four files: edge.1.txt, branch_length.1.txt, species_name.1.txt, chromosomeID.synteny.txt, and path_list.txt. Please follow the descriptions of the example input files to prepare the input files for your own study. Please keep the input files in the directory specified by the argument '-p' (or '--root_path'). The directory of the input data files are set to be the current working directory by default. For the current version of Phylo-HMRF, please use the same file names as used in the example.

- edge.1.txt describes the topolopy of the phylogenetic tree. Phylo-HMRF uses binary trees. Please represent the phylogenetic tree in your study as a binary tree by showing the connectivity between a node and each of its children nodes. Please also add an initial ancestor node as the remote root node of the tree, in addition to the root node that exists in the phylogenetic tree of the clade of studied species. The nodes, including the remote root node, are numbered in the order of up-to-down and left-to-right and the index starts from 0. Therefore, node 0 represents the remote root node, and node 1 represents the root node of the phylogenetic tree of the clade of studied species. Each row of the edge.1.txt shows the indices of a pair of nodes, of which the first node is the parent node and the second node is one of its children. The provide example uses a phylogenetic tree with five leaf nodes (observed species).

- branch.length.1.txt shows the length of each branch of the phylogenetic tree (including the remote root node). The branches are numbered in the order of up-to-down and left-to-right, and the index starts from 0. The branch lengths are not used in the current released functions of the Phylo-HMRF, which infer equivalent transformed branch lengths that combine the temporal evolution time with the evolutionary rate or parameters (e.g., selection strength, Brownian-motion fluctuation) along this branch. Please use any nonnegative real values for the branch lengths if they are unknown to you. The branch lengths may be used in some of the functions of further improved Phylo-HMRF. 

- species_name.1.txt shows the names of species in the study. Each row of species_name.1.txt shows the species name or the corresponding genome assembly name of one species in the study. 

- chromosomeID.synteny.txt shows the syntenic regions on the chromosome where Phylo-HMRF is applied. Each row of the chromosomeID.synteny.txt shows the start coordinate, the stop coordinate, and the length of one syntenic region, respectively, delimited by a tab character.

- path_list.txt shows the directory where the Hi-C contact frequency files of each species are kept. Each row of the path_list.txt shows the directory of the Hi-C contact frequency files of one species. The dictories for the species are in the same order with the corresponding species names in the species_name.1.txt. The Hi-C contact files of one species include the Hi-C contact frequency file on each chromsome (or some chromsomes) of the corresponding species. Each row of the Hi-C contact frequency file shows the normalized Hi-C contact frequency between a pair of genome loci on the corresponding chromosome. The normalized Hi-C contact frequency file on a chromosome could be extracted from the raw genome-wide Hi-C contact frequency file of the species using tools such as the Juicer Tools. Please name the Hi-C contact frequency file on each chromsome of a species as chromosomeID.(Resolution/1000)K.txt, e.g., chr1.50K.txt, where the chromosome is chr1 and the resolution (size of a unit genome region) is 50Kb.

************************************************************************************
# Required pre-installed packages
Phylo-HMRF requires the following packages to be installed:
- Python (tested on version 2.7.12)
- scikit-learn (tested on version 0.18)
- NumPy (tested on version 1.15.2)
- SciPy (tested on version 0.18.0)
- pandas (tested on version 0.18.1)
- Python wrapper for GCO library: pygco (Please refer to https://github.com/yujiali/pygco to install pygco)

You could install the Anaconda (avilable from https://www.continuum.io/downloads) for convenience, which provides a open souce collection of widely used data science packages including Python and NumPy.


