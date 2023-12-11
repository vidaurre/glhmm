# This Python code is a translation of part of the PALM (Permutation Analysis of Linear Models) package, originally developed by Anderson M. Winkler. 
# PALM is a powerful tool for permutation-based statistical analysis.

# To learn more about PALM, please visit the official PALM website:
# http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM

# In this Python translation, our primary focus is on accommodating family structure within your dataset. 
# The techniques employed in PALM for managing exchangeability blocks are detailed in the following publication:

# Title: Multi-level block permutation
# Authors: Winkler AM, Webster MA, Vidaurre D, Nichols TE, Smith SM.
# Source: Neuroimage. 2015;123:253-68 (Open Access)
# DOI: 10.1016/j.neuroimage.2015.05.092

# Translated by:
# Nick Y. Larsen
# CFIN / Aarhus University
# Sep/2023 (first version)

# We would like to acknowledge Anderson M. Winkler and all contributors to the PALM package for their valuable work in the field of permutation analysis.

import numpy as np
import pandas as pd
######################### PART 0 - hcp2block #########################################################
def hcp2block(file, blocksfile=None, dz2sib=False, ids=None):
    """
    Convert HCP-style twin data into block structure.

    Parameters:
    --------------
        file (str): Path to the input CSV file containing twin data.
        blocksfile (str, optional): Path to save the resulting blocks as a CSV file.
        dz2sib (bool, optional): If True, handle non-monozygotic twins as siblings. Default is False.
        ids (list or array-like, optional): List of subject IDs to include. Default is None.

    Returns:
    ----------  
        tuple: A tuple containing three elements:
            tab (numpy.ndarray): A modified table of twin data.
            B (numpy.ndarray): Block structure representing relationships between subjects.
            famtype (numpy.ndarray): An array indicating the type of each family.
    """
    # Load data
    tmp = pd.read_csv(file)
    
    # Handle missing zygosity
    if 'Zygosity' not in tmp.columns:
        tmp['Zygosity'] = np.where(tmp['ZygosityGT'].isna() | (tmp['ZygosityGT'] == ' ') | tmp['ZygosityGT'].isnull(),
                                    tmp['ZygositySR'], tmp['ZygosityGT'])

    # Select columns of interest
    cols = ['Subject', 'Mother_ID', 'Father_ID', 'Zygosity', 'Age_in_Yrs']
    tab = tmp[cols].copy()
    age = tmp['Age_in_Yrs']

    # Remove subjects with missing data
    tab_nan = (tab.iloc[:, 0:3].isna() | tab['Zygosity'].isna() | tab['Age_in_Yrs'].isna()).any(axis=1)
    # Remove missing data
    tab_empty =(tab== ' ').any(axis=1)
    # clean up table
    tab0 = tab_nan+tab_empty
    idstodel = tab.loc[tab0, 'Subject'].tolist()


    if len(idstodel) > 0:
        print(f"These subjects have data missing and will be removed: {idstodel}")

    # Create new table
    tab = tab[~tab0]
    age = age[~tab0]
    N = tab.shape[0] 

    # Handle non-monozygotic twins
    if dz2sib:
        for n in range(N):
            if tab.iloc[n, 3].lower() in ['notmz', 'dz']:
                tab.iloc[n, 3] = 'NotTwin'

    # Convert zygosity strings to identifiers
    sibtype = np.zeros(N, dtype=int)
    for n in range(N):
        if tab.iloc[n, 3].lower() == 'nottwin':
            sibtype[n] = 10
        elif tab.iloc[n, 3].lower() in ['notmz', 'dz']:
            sibtype[n] = 100
        elif tab.iloc[n, 3].lower() == 'mz':
            sibtype[n] = 1000
    tab = tab.iloc[:, :3]

    # Subselect subjects
    if ids is not None:
        if isinstance(ids[0], bool):
            tab = tab[ids]
            sibtype = sibtype[ids]
        else:
            idx = np.isin(tab[:, 0].astype(int), ids)
            if np.any(~idx):
                print(f"These subjects don't exist in the file and will be removed: {ids[~idx]}")
            ids = ids[idx]
            tabnew, sibtypenew, agenew = [], [], []
            for n in ids:
                idx = tab[:, 0].astype(int) == n
                tabnew.append(tab[idx])
                sibtypenew.append(sibtype[idx])
                agenew.append(age[idx])
            tab = np.array(tabnew)
            sibtype = np.array(sibtypenew)
            age = np.array(agenew)
    N = tab.shape[0]

    # Create family IDs
    famid = np.zeros(N, dtype=int)
    U, inv_idx = np.unique(tab.iloc[:, 1:3], axis=0, return_inverse=True)
    for u in range(U.shape[0]):
        uidx = np.all(tab.iloc[:, 1:3] == U[u], axis=1)
        famid[uidx] = u
        
    # Merge families for parents belonging to multiple families
    par = tab.iloc[:, 1:3]
    for p in par.values.flatten():
        pidx = np.any(par == p, axis=1)
        #print(np.unique(famid[pidx]))
        famids = np.unique(famid[pidx])
        for f in famids:
            famid[famid == f] = famids[0]

    # Label each family type
    F = np.unique(famid)
    famtype = np.zeros(N, dtype=int)
    for f in F:
        fidx = famid == f
        famtype[fidx] = np.sum(sibtype[fidx]) + len(np.unique(tab.iloc[fidx, 1:3]))

    #famtype

    # Correction section, because there need to be more than one person to become a twin pair
    # Handle twins with missing pair data
    # Twins which pair data isn't available should be treated as
    # non-twins, so fix and repeat computing the family types
    idx = ((sibtype == 100) & (famtype >= 100) & (famtype <= 199)) | \
            ((sibtype == 1000) & (famtype >= 1000) & (famtype <= 1999))
    sibtype[idx] = 10
    for f in F:
        fidx = famid == f
        famtype[fidx] = np.sum(sibtype[fidx]) + len(np.unique(tab.iloc[fidx, 1:3]))    
        
    # Append the new info to the table.
    tab = np.column_stack((tab, sibtype,famid, famtype))
    tab
    # Combine columns famid, sibtype, and age into a single array
    #combined_array = np.column_stack((famid, sibtype, age))
    # Use lexsort to obtain the indices that would sort the combined array
    # Sort the data and obtain sorting indices
    idx = np.lexsort((age, sibtype, famid))  # This sorts by famid, then sibtype, and finally age
    idxback = np.argsort(idx)
    tab = tab[idx]
    sibtype = sibtype[idx]
    famid = famid[idx]
    famtype = famtype[idx]
    age = age.iloc[idx]    

    # Create blocks for each family
    B = []
    for f in range(len(F)):
        fidx = famid == F[f]
        ft = famtype[np.where(fidx)[0][0]]
        if ft in np.concatenate((np.arange(12, 100, 10), [23, 202, 2002])):
            B.append(np.column_stack((famid[fidx]+1, sibtype[fidx], tab[fidx, 0])))
        else:
            B.append(np.column_stack((-(famid[fidx]+1), sibtype[fidx], tab[fidx, 0])))
            if ft == 33:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 2 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 3) or \
                        (np.sum(tabx[:, 0] == tabx[s, 0]) == 3 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 2):
                        B[-1][s, 1] += 1
            elif ft == 53:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 3 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 5) or \
                        (np.sum(tabx[:, 0] == tabx[s, 0]) == 5 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 3):
                        B[-1][s, 1] += 1
        
            elif ft == 234:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 1 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 3) or \
                        (np.sum(tabx[:, 0] == tabx[s, 0]) == 3 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 1):
                        B[-1][s, 1] += 1
            elif ft == 54:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if np.sum(tabx[:, 0] == tabx[s, 0]) == 4 and np.sum(tabx[:, 1] == tabx[s, 1]) == 2:
                        B[-1][s, 1] += 1
                    elif np.sum(tabx[:, 0] == tabx[s, 0]) == 1 and np.sum(tabx[:, 1] == tabx[s, 1]) == 3:
                        B[-1][s, 1] -= 1                          
            
            
            elif ft == 34:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 2 and
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 2):
                        B[f][s, 1] += 1
                        famtype[fidx] = 39  
                                
            elif ft == 43:
                tabx = tab[fidx, 1:3]
                k = 0
                for s in range(tabx.shape[0]):
                    if tabx[s, 0] == tabx[0, 0] and tabx[s, 1] == tabx[0, 1]:
                        B[-1][s, 1] += 1
                        k += 1
                if k == 2:
                    famtype[fidx] = 49
                    B[-1][:, 0] = -B[-1][:, 0]
            elif ft == 44:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if np.sum(tabx[:, 0] == tabx[s, 0]) == 4 and np.sum(tabx[:, 1] == tabx[s, 1]) == 2:
                        B[-1][s, 1] += 1        
                        
            elif ft == 223:
                sibx = sibtype[fidx]
                B[-1][sibx == 10, 1] = -B[-1][sibx == 10, 1]
            elif ft == 302:
                famtype[fidx] = 212
                tmpage = age[fidx]
                if tmpage.iloc[0] == tmpage.iloc[1]:
                    B[-1][2, 1] = 10
                elif tmpage.iloc[0] == tmpage.iloc[2]:
                    B[-1][1, 1] = 10
                elif tmpage.iloc[1] == tmpage.iloc[2]:
                    B[-1][0, 1] = 10
            elif ft == 313 or ft == 314:
                famtype[fidx] = ft - 100 + 10
                if (famtype[fidx] == 223).all():
                    # Identify the elements that are equal to 223 and need changing
                    to_change = (famtype == 223) & fidx
                    # Update the values at the selected indices
                    famtype[fidx] = 229

                tmpage = age[fidx]
                #didx = np.where(B[-1][:, 1] == 100)[0]
                didx = np.where(B[-1][:, 1] == 100)[0]
                if tmpage.iloc[didx[0]] == tmpage.iloc[didx[1]]:
                    B[-1][didx[2], 1] = 10
                elif tmpage.iloc[didx[0]] == tmpage.iloc[didx[2]]:
                    B[-1][didx[1], 1] = 10
                elif tmpage.iloc[didx[1]] == tmpage.iloc[didx[2]]:
                    B[-1][didx[0], 1] = 10  
                    
            # Additional case: ft == 2023
            elif ft == 2023:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if np.sum(tabx[:, 0] == tabx[s, 0]) == 4 and \
                    (np.sum(tabx[:, 1] == tabx[s, 1]) == 1 or np.sum(tabx[:, 1] == tabx[s, 1]) == 3):
                        famtype[fidx] = 2029
                        if B[f][s, 1] == 10:
                            B[f][s, 1] = -B[f][s, 1]
                            

    B = np.hstack((-np.ones((N, 1),dtype=int), famtype.reshape(-1, 1), np.concatenate(B, axis=0)))

    # Sort back to the original order
    B = B[idxback, :]
    tab = tab[idxback, :]
    tab[:, 5] = B[:, 1]


    blocksfile = "EB.csv"
    if blocksfile is not None and isinstance(blocksfile, str):
        # Save B as a CSV file with integer precision
        np.savetxt(blocksfile, B, delimiter=',', fmt='%d')
        
    # Return tab, B, and famtype
    return tab, B, famtype

######################### PART 1 - Reindix #########################################################
import numpy as np

def renumber(B):
    
    """
    Renumber the elements in a 2D numpy array B, preserving their order within distinct blocks.

    This function renumbers the elements in the input array B based on distinct values in its first column.
    Each distinct value represents a block, and the elements within each block are renumbered sequentially,
    while preserving the relative order of elements within each block.

    Parameters:
    --------------
    B (numpy.ndarray): The 2D input array to be renumbered.

    Returns:
    ----------  
    tuple: A tuple containing:
        - Br (numpy.ndarray): The renumbered array, where elements are renumbered within blocks.
        - addcol (bool): A boolean indicating whether a column was added during renumbering.

    """

    # Extract the first column of the input array B
    B1 = B[:, 0]
    # Find the unique values in B1 and store them in U
    U = np.unique(B1)
    # Create a boolean array to keep track of added columns
    addcolvec = np.zeros_like(U, dtype=bool)
    # Get the number of unique values
    nU = U.shape[0]
    # Create an empty array Br with the same shape as B
    Br = np.zeros_like(B)
    
    # Loop through unique values in B1
    for u in range(nU):
         # Find indices where B1 is equal to the current unique value U[u]
        idx = B1 == U[u]
        # Renumber the corresponding rows in Br based on the index
        Br[idx, 0] = (u + 1) * np.sign(U[u])
        
        # Check if B has more than one column
        if B.shape[1] > 1:
            # Recursively call renumber for the remaining columns and update addcolvec
            Br[idx, 1:], addcolvec[u] = renumber(B[idx, 1:])
        elif np.sum(idx) > 1:
             # If there's only one column and more than one matching row, set addcol to True
            addcol = True
            Br[idx] = -np.abs(B[idx])
        else:
            addcol = False
    # Check if B has more than one column and if any columns were added
    if B.shape[1] > 1:
        addcol = np.any(addcolvec)
    # Return the renumbered array Br and the addcol flag
    return Br, addcol

def palm_reindex(B, meth='fixleaves'):
    """
    Reindex a 2D numpy array using different procedures while preserving block structure.

    This function reorders the elements of a 2D numpy array `B` by applying one of several reindexing methods.
    The primary goal of reindexing is to assign new values to elements in such a way that they are organized
    in a desired order or structure.

    Parameters:
    --------------
    B (numpy.ndarray): The 2D input array to be reindexed.
    meth (str, optional): The reindexing method to be applied. It can take one of the following values:
        - 'fixleaves': This method reindexes the input array by preserving the order of unique values in the
          first column and recursively reindexes the remaining columns. It is well-suited for hierarchical
          data where the first column represents levels or leaves.
        - 'continuous': This method reindexes the input array by assigning new values to elements in a
          continuous, non-overlapping manner within each column. It is useful for continuous data or when
          preserving the order of unique values is not a requirement.
        - 'restart': This method reindexes the input array by restarting the numbering from 1 for each block
          of unique values in the first column. It is suitable for data that naturally breaks into distinct
          segments or blocks.
        - 'mixed': This method combines both the 'fixleaves' and 'continuous' reindexing methods. It reindexes
          the first columns using 'fixleaves' and the remaining columns using 'continuous', creating a mixed
          reindexing scheme.

    Returns:
    ----------  
    numpy.ndarray: The reindexed array, preserving the block structure based on the chosen method.


    Raises:
    ValueError: If the `meth` parameter is not one of the valid reindexing methods.
    """

    # Convert meth to lowercase
    meth = meth.lower()
    
    # Initialize the output array Br with zeros
    Br = np.zeros_like(B)
    
    if meth == 'continuous':
        # Find unique values in the first column of B
        U = np.unique(B[:, 0])
        
        # Renumber the first column based on unique values
        for u in range(U.shape[0]):
            idx = B[:, 0] == U[u]
            Br[idx, 0] = (u + 1) * np.sign(U[u])
        
        # Loop through columns starting from the 2nd column    
        for b in range(1, B.shape[1]):  # From the 2nd column onwards
            Bb = B[:, b]
            Bp = Br[:, b - 1]  # Previous column
            # Find unique values in the previous column
            Up = np.unique(Bp)
            cnt = 1
            
            # Renumber elements within blocks based on unique values
            for up in range(Up.shape[0]):
                idxp = Bp == Up[up]
                U = np.unique(Bb[idxp])
                
                # Renumber elements within the block
                for u in range(U.shape[0]):
                    idx = np.logical_and(Bb == U[u], idxp)
                    Br[idx, b] = cnt * np.sign(U[u])
                    cnt += 1
                    
    elif meth == 'restart':
        # Renumber each block separately, starting from 1
        Br, _ = renumber(B)
        
    elif meth == 'mixed':
        # Mix both 'restart' and 'continuous' methods
        Ba, _ = palm_reindex(B, 'restart')
        Bb, _ = palm_reindex(B, 'continuous')
        Br = np.hstack((Ba[:, :-1], Bb[:, -1:]))
        
    elif meth=="fixleaves":
        # Reindex using 'fixleaves' method as defined in the renumber function

        B1 = B[:, 0]
        U = np.unique(B1)
        addcolvec = np.zeros_like(U, dtype=bool)
        nU = U.shape[0]
        Br = np.zeros_like(B)

        for u in range(nU):
            idx = B1 == U[u]
            Br[idx, 0] = (u + 1) * np.sign(U[u])

            if B.shape[1] > 1:
                Br[idx, 1:], addcolvec[u] = renumber(B[idx, 1:])
            elif np.sum(idx) > 1:
                addcol = True
                Br[idx] = -np.abs(B[idx])
            else:
                addcol = False

        if B.shape[1] > 1:
            addcol = np.any(addcolvec)
        
        if addcol:
            # Add a column of sequential numbers to Br and reindex
            col = np.arange(1, Br.shape[0] + 1).reshape(-1, 1)
            Br = np.hstack((Br, col))
            Br, _ = renumber(Br)
            
    else:
        # Raise a ValueError for an unknown method
        raise ValueError(f'Unknown method: {meth}')
    # Return the reindexed array Br
    return Br


######################### PART 2 - PALMTREE #########################################################
import numpy as np
import copy
from palm_maxshuf import palm_maxshuf,is_single_value

def palm_permtree(Ptree, nP, CMC=False, maxP=None):
    """
    Generate permutations of a given palm tree structure.

    This function generates permutations of a palm tree structure represented by Ptree. Permutations are created by
    shuffling the branches of the palm tree. The number of permutations is controlled by the 'nP' parameter.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure to be permuted.
    nP (int): The number of permutations to generate.
    CMC (bool, optional): Whether to use Combinatorial Monte Carlo (CMC) method for permutation.
                          Defaults to False.
    maxP (int, optional): The maximum number of permutations allowed. If not provided, it is calculated automatically.

    Returns:
    ----------  
    numpy.ndarray: An array representing the permutations. Each row corresponds to a permutation, with the first
                   column always representing the identity permutation.

    Note:
    - If 'CMC' is False and 'nP' is greater than 'maxP' / 2, a warning message is displayed, as it may take a
      considerable amount of time to find non-repeated permutations.
    - The function utilizes the 'pickperm' and 'randomperm' helper functions for the permutation process.
    """
    
    if nP == 1 and not maxP:
        # Calculate the maximum number of permutations if not provided
        maxP = palm_maxshuf(Ptree, 'perms')
        if nP > maxP:
            nP = maxP  # The cap is only imposed if maxP isn't supplied
    
    
    # Permutation #1 is no permutation, regardless.
    P = pickperm(Ptree, np.array([], dtype=int))
    P = np.hstack((P.reshape(-1,1), np.zeros((P.shape[0], nP - 1), dtype=int)))
    
    
    # Generate all other permutations up to nP
    if nP == 1:
        pass
    elif CMC or nP > maxP:
        for p in range(2, nP + 1):
            Ptree_perm = copy.deepcopy(Ptree)
            Ptree_perm = randomperm(Ptree_perm)
            P[:, p - 1] = pickperm(Ptree_perm, [])
    else:
        if nP > maxP / 2:
            # Inform the user about the potentially long runtime
            print(f'The maximum number of permutations ({maxP}) is not much larger than\n'
                  f'the number you chose to run ({nP}). This means it may take a while (from\n'
                  f'a few seconds to several minutes) to find non-repeated permutations.\n'
                  'Consider instead running exhaustively all possible permutations. It may be faster.')
        for p in range(1, nP):
            whiletest = True
            while whiletest:
                Ptree_perm = copy.deepcopy(Ptree)
                Ptree_perm = randomperm(Ptree_perm)
                P[:, p] = pickperm(Ptree_perm, [])
                
                whiletest = np.any(np.all(P[:, :p] == P[:, p][:, np.newaxis], axis=0))
    
    # The grouping into branches screws up the original order, which
    # can be restored by noting that the 1st permutation is always
    # the identity, so with indices 1:N. This same variable idx can
    # be used to likewise fix the order of sign-flips (separate func).
    idx = np.argsort(P[:, 0])
    P = P[idx, :]
    
    return P

def pickperm(Ptree, P):
    """
    Extract a permutation from a palm tree structure.

    This function extracts a permutation from a given palm tree structure. It does not perform the permutation
    but returns the indices representing the already permuted tree.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    P (numpy.ndarray): The current state of the permutation.

    Returns:
    ----------  
    numpy.ndarray: An array of indices representing the permutation of the palm tree structure.
    """
    # Check if Ptree is a list and has three elements, then recursively call pickperm on the third element
    if isinstance(Ptree,list):
        if len(Ptree) == 3:
            P = pickperm(Ptree[2],P)
    # Check if the shape of Ptree is (N, 3), where N is the number of branches
    elif Ptree.shape[1] ==3:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Recursively call pickperm on the third element of the branch
            P = pickperm(Ptree[u][2],P)
    # Check if the shape of Ptree is (N, 1)
    elif Ptree.shape[1] ==1:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Concatenate the first element of the branch (a submatrix) to P
            P = np.concatenate((P, Ptree[u][0]), axis=None)
    return P

def randomperm(Ptree_perm):
    """
    Create a random permutation of a palm tree structure.

    This function generates a random permutation of a given palm tree structure by shuffling its branches.

    Parameters:
    --------------
    Ptree_perm (list or numpy.ndarray): The palm tree structure to be permuted.

    Returns:
    ----------  
    list: The randomly permuted palm tree structure.
    """
    # Check if Ptree_perm is a list and has three elements, then recursively call randomperm on the third element
    if isinstance(Ptree_perm,list):
        if len(Ptree_perm) == 3:
            Ptree_perm = randomperm(Ptree_perm[2])
            
        
    # Get the number of branches in Ptree_perm
    nU = Ptree_perm.shape[0]
    # Loop through each branch
    for u in range(nU):
        # Check if the first element of the branch is a single value and not NaN
        if is_single_value(Ptree_perm[u][0]):
            if not np.isnan(Ptree_perm[u][0]):
                tmp = 1
                # Shuffle the first element of the branch
                np.random.shuffle(Ptree_perm[u][0])
                # Check if tmp is not equal to the first element of the branch
                if np.any(tmp != Ptree_perm[u][0][0]):
                     # Rearrange the third element of the branch based on the shuffled indices
                    Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :] 
        # Check if the first element of the branch is a list with three elements            
        elif isinstance(Ptree_perm[u][0],list) and len(Ptree_perm[u][0])==3:
            tmp = 1
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]
            
        else:
            tmp = np.arange(1,len(Ptree_perm[u][0][:,0])+1,dtype=int)
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][:, 0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2] =Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]      
            
        # Make sure the next isn't the last level.
        if Ptree_perm[u][2].shape[1] > 1:
            # Recursively call randomperm on the third element of the branch
            Ptree_perm[u][2] = randomperm(Ptree_perm[u][2])  
            
    return Ptree_perm


######################### PART 3.1 - Permute PTREE #########################################################
#### Permutation functions
import numpy as np

def palm_maxshuf(Ptree, stype='perms', uselog=False):
    """
    Calculate the maximum number of shufflings (permutations or sign-flips) for a given palm tree structure.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    stype (str, optional): The type of shuffling to calculate ('perms' for permutations by default).
    uselog (bool, optional): A flag indicating whether to calculate using logarithmic values (defaults to False).

    Returns:
    ----------  
    int: The maximum number of shufflings (permutations or sign-flips) based on the specified criteria.
    """
    
    # Calculate the maximum number of shufflings based on user-defined options
    if uselog:
        if stype == 'perms':
            maxb = lmaxpermnode(Ptree, 0)
    
    else:
        if stype == 'perms':
            maxb = maxpermnode(Ptree, 1)
    return maxb

def maxpermnode(Ptree, np):
    """
    Calculate the maximum number of permutations within a palm tree node.

    This function recursively calculates the maximum number of permutations within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    np (int): The current number of permutations (initialized to 1).

    Returns:
    ----------  
    int: The maximum number of permutations within the node.
    """
    for u in range(len(Ptree)):
        n_p = n_p * seq2np(Ptree[u][0][:, 0])
        if len(Ptree[u][2][0]) > 1:
            n_p = maxpermnode(Ptree[u][2], np)
    return n_p

def seq2np(S):
    """
    Calculate the number of permutations for a given sequence.

    This function calculates the number of permutations for a given sequence.

    Parameters:
    --------------
    S (numpy.ndarray): The input sequence.

    Returns:
    ----------  
    int: The number of permutations for the sequence.
    """
    U, cnt = np.unique(S, return_counts=True)
    n_p = np.math.factorial(len(S)) / np.prod(np.math.factorial(cnt))
    return n_p

def maxflipnode(Ptree, ns):
    """
    Calculate the maximum number of sign-flips within a palm tree node.

    This function recursively calculates the maximum number of sign-flips within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    ns (int): The current number of sign-flips (initialized to 1).

    Returns:
    ----------  
    int: The maximum number of sign-flips within the node.
    """
    for u in range(len(Ptree)):
        if len(Ptree[u][2][0]) > 1:
            ns = maxflipnode(Ptree[u][2], ns)
        ns = ns * (2 ** len(Ptree[u][1]))
    return ns

def lmaxpermnode(Ptree, n_p):
    """
    Calculate the logarithm of the maximum number of permutations within a palm tree node.

    This function calculates the logarithm of the maximum number of permutations within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    n_p (int): The current logarithm of permutations (initialized to 0).

    Returns:
    ----------  
    int: The logarithm of the maximum number of permutations within the node.
    """
    if isinstance(Ptree,list):
        n_p = n_p + lseq2np(Ptree[0])
        if Ptree[2].shape[1] > 1:
            n_p = lmaxpermnode(Ptree[2], n_p)
    else:
        for u in range(Ptree.shape[0]):
            if isinstance(Ptree[u][0],list):
                n_p = n_p + lseq2np(Ptree[u][0][0])
                if Ptree[u][2].shape[1] > 1:
                    #n_p = lmaxpermnode(Ptree[u][2][0][2], n_p)
                    n_p = lmaxpermnode(Ptree[u][2], n_p)
            elif is_single_value(Ptree[u][0]):
                n_p = n_p + lseq2np(Ptree[u][0])
                
                if len(Ptree[u]) > 2 and Ptree[u][2].shape[1] > 1:
                    n_p = lmaxpermnode(Ptree[u][2], n_p) 
            else:     
                n_p = n_p + lseq2np(Ptree[u][0][:,0])
                
                if len(Ptree[u]) > 2 and Ptree[u][2].shape[1] > 1:
                    n_p = lmaxpermnode(Ptree[u][2], n_p) 
    
    return n_p

def lseq2np(S):
    """
    Calculate the logarithm of the number of permutations for a given sequence.

    This function calculates the logarithm of the number of permutations for a given sequence.

    Parameters:
    --------------
    S (numpy.ndarray): The input sequence.

    Returns:
    ----------  
    int: The logarithm of the number of permutations for the sequence.
    """
    if is_single_value(S):
        nS = 1
        if np.isnan(S):
            U = np.nan
            cnt = 0
        else:
            U, cnt = np.unique(S, return_counts=True)
            
    else: 
        nS = len(S)
        U, cnt = np.unique(S, return_counts=True)
        
    #lfac=palm_factorial(nS)
    lfac=palm_factorial()
    n_p = lfac[nS] - np.sum(lfac[cnt])
    return n_p

def lmaxflipnode(Ptree, ns):
    """
    Calculate the logarithm of the maximum number of sign-flips within a palm tree node.

    This function calculates the logarithm of the maximum number of sign-flips within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    ns (int): The current logarithm of sign-flips (initialized to 0).

    Returns:
    ----------  
    int: The logarithm of the maximum number of sign-flips within the node.
    """
    for u in range(len(Ptree)):
        if len(Ptree[u][2][0]) > 1:
            ns = lmaxflipnode(Ptree[u][2], ns)
        ns = ns + len(Ptree[u][1])
    return ns

def is_single_value(variable):
    """
    Check if an array contains a singlevalue.

    This function checks if an array contains a singlevalue.

    Parameters:
    --------------
    arr (numpy.ndarray or list): The array to be checked.

    Returns:
    ----------  
    bool: True if the array contains a single value, False otherwise.
    """
    return isinstance(variable, (int, float, complex))

def palm_factorial(N=101):
    """
    Calculate logarithmically scaled factorials up to a given number.

    This function precomputes logarithmically scaled factorials up to a specified number.

    Parameters:
    --------------
    N (int, optional): The maximum number for which to precompute factorials (defaults to 101).

    Returns:
    ----------  
    numpy.ndarray: An array of precomputed logarithmically scaled factorials.
    """
    if N == 1:
        N = 101
    # Initialize the lf array with zeros
    lf = np.zeros(N+1)

    # Calculate log(factorial) values
    for n in range(1, N+1):
        lf[n] = np.log(n) + lf[n-1]

    return lf

######################### PART 3.2 - PALMTREE #########################################################
#### Permute PALM tree
import numpy as np
import copy

def palm_permtree(Ptree, nP, CMC=False, maxP=None):
    """
    Generate permutations of a given palm tree structure.

    This function generates permutations of a palm tree structure represented by Ptree. Permutations are created by
    shuffling the branches of the palm tree. The number of permutations is controlled by the 'nP' parameter.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure to be permuted.
    nP (int): The number of permutations to generate.
    CMC (bool, optional): Whether to use Combinatorial Monte Carlo (CMC) method for permutation.
                          Defaults to False.
    maxP (int, optional): The maximum number of permutations allowed. If not provided, it is calculated automatically.

    Returns:
    ----------  
    numpy.ndarray: An array representing the permutations. Each row corresponds to a permutation, with the first
                   column always representing the identity permutation.

    Note:
    - If 'CMC' is False and 'nP' is greater than 'maxP' / 2, a warning message is displayed, as it may take a
      considerable amount of time to find non-repeated permutations.
    - The function utilizes the 'pickperm' and 'randomperm' helper functions for the permutation process.
    """
    
    if nP == 1 and not maxP:
        # Calculate the maximum number of permutations if not provided
        maxP = palm_maxshuf(Ptree, 'perms')
        if nP > maxP:
            nP = maxP  # The cap is only imposed if maxP isn't supplied
    
    
    # Permutation #1 is no permutation, regardless.
    P = pickperm(Ptree, np.array([], dtype=int))
    P = np.hstack((P.reshape(-1,1), np.zeros((P.shape[0], nP - 1), dtype=int)))
    
    
    # Generate all other permutations up to nP
    if nP == 1:
        pass
    elif CMC or nP > maxP:
        for p in range(2, nP + 1):
            Ptree_perm = copy.deepcopy(Ptree)
            Ptree_perm = randomperm(Ptree_perm)
            P[:, p - 1] = pickperm(Ptree_perm, [])
    else:
        if nP > maxP / 2:
            # Inform the user about the potentially long runtime
            print(f'The maximum number of permutations ({maxP}) is not much larger than\n'
                  f'the number you chose to run ({nP}). This means it may take a while (from\n'
                  f'a few seconds to several minutes) to find non-repeated permutations.\n'
                  'Consider instead running exhaustively all possible permutations. It may be faster.')
        for p in range(1, nP):
            whiletest = True
            while whiletest:
                Ptree_perm = copy.deepcopy(Ptree)
                Ptree_perm = randomperm(Ptree_perm)
                P[:, p] = pickperm(Ptree_perm, [])
                
                whiletest = np.any(np.all(P[:, :p] == P[:, p][:, np.newaxis], axis=0))
    
    # The grouping into branches screws up the original order, which
    # can be restored by noting that the 1st permutation is always
    # the identity, so with indices 1:N. This same variable idx can
    # be used to likewise fix the order of sign-flips (separate func).
    idx = np.argsort(P[:, 0])
    P = P[idx, :]
    
    return P

def pickperm(Ptree, P):
    """
    Extract a permutation from a palm tree structure.

    This function extracts a permutation from a given palm tree structure. It does not perform the permutation
    but returns the indices representing the already permuted tree.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    P (numpy.ndarray): The current state of the permutation.

    Returns:
    ----------  
    numpy.ndarray: An array of indices representing the permutation of the palm tree structure.
    """
    # Check if Ptree is a list and has three elements, then recursively call pickperm on the third element
    if isinstance(Ptree,list):
        if len(Ptree) == 3:
            P = pickperm(Ptree[2],P)
    # Check if the shape of Ptree is (N, 3), where N is the number of branches
    elif Ptree.shape[1] ==3:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Recursively call pickperm on the third element of the branch
            P = pickperm(Ptree[u][2],P)
    # Check if the shape of Ptree is (N, 1)
    elif Ptree.shape[1] ==1:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Concatenate the first element of the branch (a submatrix) to P
            P = np.concatenate((P, Ptree[u][0]), axis=None)
    return P

def randomperm(Ptree_perm):
    """
    Create a random permutation of a palm tree structure.

    This function generates a random permutation of a given palm tree structure by shuffling its branches.

    Parameters:
    --------------
    Ptree_perm (list or numpy.ndarray): The palm tree structure to be permuted.

    Returns:
    ----------  
    list: The randomly permuted palm tree structure.
    """
    # Check if Ptree_perm is a list and has three elements, then recursively call randomperm on the third element
    if isinstance(Ptree_perm,list):
        if len(Ptree_perm) == 3:
            Ptree_perm = randomperm(Ptree_perm[2])
            
        
    # Get the number of branches in Ptree_perm
    nU = Ptree_perm.shape[0]
    # Loop through each branch
    for u in range(nU):
        # Check if the first element of the branch is a single value and not NaN
        if is_single_value(Ptree_perm[u][0]):
            if not np.isnan(Ptree_perm[u][0]):
                tmp = 1
                # Shuffle the first element of the branch
                np.random.shuffle(Ptree_perm[u][0])
                # Check if tmp is not equal to the first element of the branch
                if np.any(tmp != Ptree_perm[u][0][0]):
                     # Rearrange the third element of the branch based on the shuffled indices
                    Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :] 
        # Check if the first element of the branch is a list with three elements            
        elif isinstance(Ptree_perm[u][0],list) and len(Ptree_perm[u][0])==3:
            tmp = 1
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]
            
        else:
            tmp = np.arange(1,len(Ptree_perm[u][0][:,0])+1,dtype=int)
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][:, 0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2] =Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]      
            
        # Make sure the next isn't the last level.
        if Ptree_perm[u][2].shape[1] > 1:
            # Recursively call randomperm on the third element of the branch
            Ptree_perm[u][2] = randomperm(Ptree_perm[u][2])  
            
    return Ptree_perm


######################### PART 3.3 - Permute PTREE #########################################################
###### Function that integrates the above functions in part 3
import warnings
import numpy as np
from palm_permtree import palm_permtree
from palm_maxshuf import palm_maxshuf

def palm_shuftree(Ptree,nP,CMC= False,EE = True):
    """
    Generate a set of shufflings (permutations or sign-flips) for a given palm tree structure.

    Parameters:
    --------------
    Ptree (list): The palm tree structure.
    nP (int): The number of permutations to generate.

    CMC (bool, optional): A flag indicating whether to use the Constrained Monte Carlo method (CMC).
                          Defaults to False.
    EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                        Defaults to True.

    Returns:
    ----------  
    list: A list containing the generated shufflings (permutations).
    """ 
    
    # Maximum number of shufflings (perms, sign-flips, or both)
    maxP = 1
    maxS = 1
    if EE:
        lmaxP = palm_maxshuf(Ptree, 'perms', True)
        maxP = np.exp(lmaxP)

        if np.isinf(maxP):
            print('Number of possible permutations is exp({}).'.format(lmaxP))
        else:
            print('Number of possible permutations is {}.'.format(maxP))


    maxB = maxP * maxS

    # String for the screen output below
    whatshuf = 'permutations only'
    whatshuf2 = 'perms'

    # Generate the Pset and Sset
    Pset = []
    Sset = []

    if nP == 0 or nP >= maxB:
        # Run exhaustively if the user requests too many permutations.
        # Note that here CMC is irrelevant.
        print('Generating {} shufflings ({}).'.format(maxB, whatshuf))
        if EE:
            Pset = palm_permtree(Ptree, int(round(maxP)) if maxP != np.inf else maxP, [], int(round(maxP)) if maxP != np.inf else maxP)


    elif nP < maxB:
        # Or use a subset of possible permutations.
        print('Generating {} shufflings ({}).'.format(nP, whatshuf))
        if EE:
            if nP >= maxP:
                Pset = palm_permtree(Ptree, int(round(maxP)) if maxP != np.inf else maxP, CMC, int(round(maxP)) if maxP != np.inf else maxP)
                
            else:
                Pset = palm_permtree(Ptree, nP, CMC, int(round(maxP)) if maxP != np.inf else maxP)

    return Pset
######################### PART 4 - quick_perm #########################################################

from palm_reindex import palm_reindex 
from palm_tree import palm_tree
from palm_shuftree import palm_shuftree

def palm_quickperms(EB, M=None, nP=1000, CMC=False, EE=True):
    """
    Generate a set of permutations for a given input matrix using palm methods.

    Parameters:
    --------------
    EB (numpy.ndarray): The input matrix or data.
    M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                Defaults to None.
    nP (int): The number of permutations to generate.
    CMC (bool, optional): A flag indicating whether to use the Constrained Monte Carlo method (CMC).
                          Defaults to False.
    EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                        Defaults to True.

    Returns:
    ----------  
    list: A list containing the generated permutations.
    """
    
    # Reindex the input matrix for palm methods with 'fixleaves'
    EB2 = palm_reindex(EB, 'fixleaves')
    
    # Generate a palm tree structure from the reindexed matrix
    Ptree = palm_tree(EB2)
    
    # Generate a set of shufflings (permutations) based on the palm tree structure
    Pset = palm_shuftree(Ptree, nP, CMC, EE)
    
    return Pset