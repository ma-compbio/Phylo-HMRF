# Phylo-HMRF

Phylogenetic Hidden Markov Random Field model

The command to use Phylp-HMRF for evolutionary state estimation is as follows. 

python phylo_hmrf.py [Options]

The options:

- -n, --num_states : the number of states to estimate for Phylo-HMRF, default = 10

- -p, --root_path : the directory where the input data files are kept, default = '.' (the current working directory)

- -f, --chromosome : the name of the chromosome to estimate hidden states using Phylo-HMRF, default = 1

- --num_neighbor : the number of neighbors of each node in the neighborhood system, default = 10

- -g, --estimate_type : the choice to consider edge weights in pairwise potential: 0: not consider edge weights; 3: consider edge weights, default = 3

- -b, --beta : the pairwise potentional parameter beta_0, default = 1

- --beta1 : the pairwise potential parameter beta_1, default = 0.1

- -r, --run_id : experiment id, default = 0

- -c, --cons_param : L2-norm regularization parameter, default = 4

- -i, --initial_weight : weight of initial parameters estimatd from K-means clustering for the parametr initialization in each maximization step, default = 0.5

- -j, --initial_magnitude : initial magnitude for initial parameters, default = 1

- -d, --initial_mode : initialization mode: 0: initial paramater values are non-negative; 1: initial selection strengths and Brownian motion intensities are non-negative; initial optimal values can be negative, default = 0

- --reload : whether to reload existing processed data: 1: reload; 0: not reload, default=0

- --quantile : whether to compute signal quantiles on each chromosome: 0: load existing file; 1: compute, default = 1

- --miter : max number of iterations to perform, default =60

- --resolution : genomic bin size (unit: base pair), default = 50000

- --ref_species : genome assembly ID of the reference species, default = hg38

- --chromvec : chromosomes to perform estimation: -1: all the chromosomes for human, default = 1; To perform state estimation on multiple chromosomes, please use the format: chromID_1,chromID_2,...chromID_M, where the chromosome IDs are separated by commas.

- --output : output directory to save files, default = "." (the output files are saved in the current directory)

Example: 

python phylo_hmrf.py -n 20 -r 1 --reload 0 --chromvec 21,22 --miter 100 (using Phylo-HMRF to estimate 20 states on syntenic regions on chromosome 21 and chromosome 22 jointly)
    
The input files include: edge.1.txt, branch_length.1.txt, species_name.1.txt, chromosomeID.synteny.txt, path_list.txt, chromosome size file of the reference genome, and the aligned Hi-C contact files of studied species. Please follow the descriptions of the input files to prepare the input files for your own study. Please keep the input files in the directory specified by the argument '-p' (or '--root_path'). The directory of the input data files are set to be the current working directory by default. For the current version of Phylo-HMRF, please use the same file names as shown in the descriptions. We provide some example input files in the file folder example_input. Please see the examples for the input format. 

- edge.1.txt describes the topolopy of the phylogenetic tree. Phylo-HMRF uses binary trees. Please represent the phylogenetic tree in your study as a binary tree by showing the connectivity between a node and each of its children nodes. Please also add an initial ancestor node as the remote root node of the tree, in addition to the root node that exists in the phylogenetic tree of the clade of studied species. The nodes, including the remote root node, are numbered in the order of up-to-down and left-to-right and the index starts from 0. Therefore, node 0 represents the remote root node, and node 1 represents the root node of the phylogenetic tree of the clade of studied species. Each row of the edge.1.txt shows the indices of a pair of nodes, of which the first node is the parent node and the second node is one of its children. The provide example uses a phylogenetic tree with five leaf nodes (observed species).

- branch.length.1.txt shows the length of each branch of the phylogenetic tree (including the remote root node). The branches are numbered in the order of up-to-down and left-to-right, and the index starts from 0. The branch lengths are not used in the current released functions of the Phylo-HMRF, which infer equivalent transformed branch lengths that combine the temporal evolution time with the evolutionary rate or parameters (e.g., selection strength, Brownian-motion fluctuation) along this branch. Please use any nonnegative real values for the branch lengths if they are unknown to you. The branch lengths may be used in functions of further improved Phylo-HMRF. 

- species_name.1.txt shows the names of species in the study. Each row of species_name.1.txt shows the species name or the corresponding genome assembly name of one species in the study, in the order of their occurrences as leaf nodes from left to right in the species tree.

- chromosomeID.synteny.txt shows the syntenic regions on the chromosome where Phylo-HMRF is applied. Each row of the chromosomeID.synteny.txt shows the start coordinate, the stop coordinate, and the length of one syntenic region, respectively, delimited by a tab character.

- path_list.txt shows the directory where the Hi-C contact frequency files of each species are kept. Each row of the path_list.txt shows the directory of the Hi-C contact frequency files of one species. The dictories for the species are in the same order with the corresponding species names in the species_name.1.txt. The Hi-C contact files of one species include the Hi-C contact frequency file on each chromsome (or some chromsomes) of the corresponding species. Each row of the Hi-C contact frequency file shows the normalized Hi-C contact frequency between a pair of genome loci on the corresponding chromosome. The normalized Hi-C contact frequency file on a chromosome could be extracted from the raw genome-wide Hi-C contact frequency file of the species using tools such as the Juicer Tools. Please name the Hi-C contact frequency file on each chromsome of a species as chromosomeID.(Resolution/1000)K.txt, e.g., chr1.50K.txt, where the chromosome is chr1 and the resolution (size of a unit genome region) is 50Kb.

- referenceSpecies.chrom.sizes shows the size of each chromosome of the genome of the reference speices which the Hi-C contacts in different species are aligned to. The file can be downloaded from the UCSC genome browser, named as genomeAssemblyID.chrom.sizes. 

Please see outputfile_description.txt for the descriptions of the output file. We also provide MATLAB code that could be used to extract the state estimation results from the output file and visualize the estimated states in the Hi-C contact map of each synteny block as a color image. Please see the code in the file folder processing.

Please comment or modify Line 387-393 in utility.py according to the species studied. In our study, we divided the large-size synteny regions on chr3 and chr6 of genome hg38 according to the chromosome arms, respectively. However, this only applies to genome hg38. If the reference genome is different, Line 387-393 need to be changed accordingly.

The first version of Phylo-HMRF is contained in the file folder phylo_hmrf_v1, which can be applied to state estimation on single chromosomes.

************************************************************************************
# Required pre-installed packages
Phylo-HMRF requires the following packages to be installed:
- Python (tested on version 2.7.15)
- scikit-learn (tested on version 0.18)
- NumPy (tested on version 1.15.2)
- SciPy (tested on version 0.18.0)
- pandas (tested on version 0.18.1)
- Python wrapper for GCO library: pygco (Please refer to https://github.com/yujiali/pygco to install pygco)
- MedPy (tested on version 0.3.0)
- scikit-image (tested on version 0.12.3)

You could install the Anaconda (avilable from https://www.continuum.io/downloads) for convenience, which provides a open souce collection of widely used data science packages including Python and NumPy.


