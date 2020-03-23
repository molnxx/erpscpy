from .auxiliary import *
import numpy as np
import atomium
import os.path

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
    ''' atomium structure class with the coordinates of the chains (only one chain in SCOPe, COPS and CATH files) '''
    structure = atomium.open(str(file))
    coords = []
    for chain in structure.model.chains():
        for res in chain:
            for atom in res.atoms():
                if (atom.name == 'CA' and atom.het.code != 'X'):
                    coords.append(atom.location)
    structure.coordinates = np.asarray(coords)
    return structure


