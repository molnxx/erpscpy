from .auxiliary import *
import numpy as np
import atomium
import os.path
import erpscpy.er

################
# fasta parser #
################

def fasta(file):
    '''returning a dictionary with headers as keys and sequences as values'''
    result = {}
    for line in open(file, 'r'):
        if line.startswith('>'):
            header = line[1:].rstrip()
            result[header] = ''
        else:
            result[header] += line.rstrip()
    return result


###################################################
# toggle for residue one-letter and 3-letter code #
###################################################

reslet = dict()
here = os.path.dirname(__file__)
for l in open(os.path.join(here, './residue_letters.txt'), 'r'):
    reslet[l[:3].upper()] = l[8].upper()
    reslet[l[8].upper()] = l[:3].upper()
    reslet[l[:3].lower()] = l[8].lower()
    reslet[l[8].lower()] = l[:3].lower()
    reslet[l[:3]] = l[8]
    reslet[l[8]] = l[:3]

def toggle_code(sequence, direction='1to3'):
    result = ''
    if direction == '1to3':
        for char in sequence:
            result += reslet[char]
    else:
        for three in chunks(sequence, 3):
            result += reslet[three]
    return reslet[key]


############################
# protein structure parser #
############################

def structure(file):
    ''' atomium structure class with the coordinates of the chains (only one chain in SCOPe, COPS and CATH files)
    and it's EigenRank Profile '''
    try:
        structure = atomium.open(str(file))
    except FileNotFoundError:
        structure = atomium.fetch(str(file))
    if structure.code == None:
        structure.id = os.path.basename(file).split('.')[0]
    else:
        structure.id = structure.code
    coord_dict = {}
    for chain in structure.model.chains():
        coords = []
        for res in chain:
            for atom in res.atoms():
                if (atom.name == 'CA' and atom.het.code != 'X'):
                    coords.append(atom.location)
        coord_dict[chain.internal_id] = np.asarray(coords)
    structure.coordinates = coord_dict
    return erpscpy.er.add_eigenrank(structure)


