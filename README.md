EigenRank Protein Structure Comparison (with PYthon)
================================================================

## Installation

PyPI package will come soon. Until then use:

    git clone https://github.com/molnxx/erpscpy.git

and import with:

    import erpscpy as er

## Usage

### Loading a structure

Load a structure file like .pdb or .cif. Parsing takes place with [__atomium__](https://github.com/samirelanduk/atomium). Should the file not exist, it will try to fetch it from the PDB.

    domain1 = er.parser.structure('6TCM.pdb')
    domain2 = er.parser.structure('6URH')

Added to the _atomium_ File object are the coordinates of each chain and the EigenRank-Profile:

    print(domain1.coordinates)
    {'A': array([[-31.092,  50.291,   2.166],....

    print(domain1.er['A'])
    [-2.7930536  -2.1671908  -1.1287678  -0.45051202 -1.2144186  -0.8865472....

### Alignment of EigenRank-Profiles

The input needs to be the _structure_ object from above. When multiple chains are present, only the profile of the first chain will be aligned. Gap cost and an added limit for the local alignment dynamic algorithm can be set (default: gap = 1 and limit = 0.7). More flexibility will come in later releases; traces of an improved scoring of (normally distributed) ER profiles can be found in the code already.

    align = er.Alignment(domain1, domain2, gap = 1, limit = 0.7)

The _Alignment_ object has many attributes for analysis. Among them are the residue indices and the score of the local alignment (*i_list*, *j_list*, *score*), rotation matrix ([Kabsch](https://en.wikipedia.org/wiki/Kabsch_algorithm)), [RMSD](https://en.wikipedia.org/wiki/Root-mean-square_deviation), [GDT_TS](https://en.wikipedia.org/wiki/Global_distance_test) and the coordinates of both aligned structures (*query_aligned*, *target_aligned*).

## Cite

Heinke, F., Hempel, L., and Labudde, D. (2019). A Novel Approach
for Fast Protein Structure Comparison and Heuristic Structure Database Searching
Based on Residue EigenRank Scores.
