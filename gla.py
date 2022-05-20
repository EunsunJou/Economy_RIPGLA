
##### Python implementation of Stochastic OT
##### Inspired by Harmonic Grammar implementation by Connor McLaughlin:
##### https://github.com/connormcl/harmonic-grammar


##### SOME TERMINOLOGY #####
# Overt (form): a datum that the learner hears. 
#               It may contain stress info but not foot structure info.
# Input (form): the underlying representation of an overt form
# Parse: the structural analysis of an overt form, including foot information.
#        It is the output form of a tableau corresponding to the input form.
#        The parse of an overt form varies depending on constranit ranking. 
# Generate: Compute the parse given the input and a constraint ranking.

##### SOME ABBREVIATIONS (FOR VARIABLE NAMING) #####
# constraint: const
# violation: viol
# dictionary: dict
# candidate: cand
# input: inp (to avoid overlapping with input() function)

import re
import random
import sys
import datetime
import os
import matplotlib.pyplot as plt
from operator import itemgetter
from labellines import labelLine

##### Part 0: Open and save grammar and target files ############################



# The Grammar file is a specific format of a txt file created by Praat
# (It is called an "otgrammar" object in the Praat documentation.

def grammar_string(txtfile):
    grammar_file = open(txtfile, 'r')
    grammar_text = grammar_file.read()
    grammar_file.close()
    return grammar_text

def grammar_readlines(txtfile):
    grammar_file = open(txtfile, 'r')
    grammar_lines = grammar_file.readlines()
    grammar_file.close()
    return grammar_lines

def read_and_rstrip(txtfile):
    target_file = open(txtfile, 'r')
    target_list = target_file.readlines()
    target_list = [x.rstrip() for x in target_list]
    target_file.close()
    return target_list

################################################################################
##### Part 1: Extract Information from Grammar File ############################
################################################################################

# Praat's otgrammar is a list of constraints followed by a list of OT tableaux, 
# which provide information about the violation profile for each input-parse pair. 
# The tableaux and its elements (input form, overt form, violation profile) 
# can be expressed in regular grammar.

### Extract list of constraints, preserving their order in grammar file
# Preserving order is important because the violation profiles in the tableaux 
# are based on this order.

### Regex Patterns
const_pattern = re.compile(r"constraint\s+\[\d+\]:\s\"(.*)\"\s*([\d\.]+)\s*")
tableau_pattern = re.compile(r"(input.*\n(\s*candidate.*\s*)+)")
input_pattern = re.compile(r"input\s+\[\d+\]:\s+\"(.*)\"") 
candidate_pattern = re.compile(r"candidate.*\[\d+\]:.*\"(.*)\"\D*([\d ]+)")
rip_pattern = re.compile(r"(\[.*\]).*(/.*/)")


# I define two helper functions here to use later in breaking up tableaux.
# This function combines two lists of the same length into a dictionary.
# The first list provides the keys, and the second list the values.
def map_lists_to_dict(keylist, valuelist):
    if len(keylist) != len(valuelist):
        raise ValueError("Length of lists do not match.")
    mapped_dict = {}
    for i in range(len(keylist)):
        mapped_dict[keylist[i]] = valuelist[i]
    return mapped_dict

# This function combines two lists into a list of tuples.
# The first list provides the 0th element of the tuple, the second list the 1st.
def map_lists_to_tuple_list(listone, listtwo):
    if len(listone) != len(listtwo):
        raise ValueError("Length of lists do not match.")
    mapped_list = []
    for i in range(len(listone)):
        mapped_list.append((listone[i], listtwo[i]))
    return mapped_list

### In order to extract information from an otgrammar,
# I extract relevant information by regex search and build a dictionary
# so that the learner can do searches across the otgrammar.
'''
Schematic structure of a tableau in Grammar File:

{input1: {overt1: {parse1: violation profile
                   parse2: violation profile
                   parse3: violation profile}
          overt2: {parse1: violation profile
                   parse2: violation profile}
          ...
          }
 input2: {overt1: {parse1: violation profile
                   parse2: violation profile}
          overt2: {parse1: violation profile
                   parse2: violation profile
                   parse3: violation profile
                   parse4: violation profile}
          ...
          } 
 input3: ...
}
'''    

### There will be two separate dictionaries: 
### one where parses are aggregated by input (input_tableaux),
### and one where parses are aggregated by overt form (overt_tableaux).

### There is only one kind of tableau to build for non-RIP GLA.
# This tableau is built using the build_tableaux function.
# (The grammar_string argument is the content of the entire otgrammar as 
# one big string, obtainable by Python's read method.)
def build_tableaux(grammar_string):
    tableaux_string = re.findall(tableau_pattern, grammar_string)
    tableaux_string = [t[0] for t in tableaux_string]
    consts = re.findall(const_pattern, grammar_string)
    consts = [c[0] for c in consts]
    
    input_tableaux = {}
    for t in tableaux_string:
        # Since there's only one input form per tableau,
        # re.findall should always yield a list of length 1)
        if len(re.findall(input_pattern, t)) == 0:
            raise ValueError("No input found in the following tableaux_string. Pleae check grammar file.\n"+t)
        elif len(re.findall(input_pattern, t)) > 1:
            raise ValueError("Found more than one input form in tableau. Please check grammar file.")

        inp = re.findall(input_pattern, t)[0]

        # Access the candidates again, to pick out parse and violation profile.
        # Each element of canddiates is a (<candidate>, <violation profile>) tuple.
        candidates = re.findall(candidate_pattern, t)
        
        # Following for-loop is identical to overt_tableaux
        parse_evals = {}
        for cand in candidates:
            parse = cand[0]
            viols_string = cand[1]
            viols = viols_string.rstrip().split(' ')
            viols = [int(x) for x in viols] 

            viol_profile = map_lists_to_dict(consts, viols)
            parse_evals[parse] = viol_profile
            
        input_tableaux[inp] = parse_evals
    
    return input_tableaux

### For RIP, the situation is more complicated.
# I need to be able to track the output corresponding with an input (i2o),
# All the parses corresponding to an input (i2p),
# And all the parses corresponding to an output (o2p).
# So I build three different kinds of tableaux using three different functions:
# build_tableaux_RIP_{i2o/i2p/o2p}.

def build_tableaux_RIP_i2o(grammar_string):
    tableaux_string = re.findall(tableau_pattern, grammar_string)
    tableaux_string = [t[0] for t in tableaux_string]
    consts = re.findall(const_pattern, grammar_string)
    consts = [c[0] for c in consts]
    
    tableaux = {}
    for t in tableaux_string:
        # Since there's only one input form per tableau,
        # re.findall should always yield a list of length 1)
        if len(re.findall(input_pattern, t)) == 0:
            raise ValueError("No input found in the following tableaux_string. Pleae check grammar file.\n"+t)
        elif len(re.findall(input_pattern, t)) > 1:
            raise ValueError("Found more than one input form in tableau. Please check grammar file.")

        inp = re.findall(input_pattern, t)[0]

        # Access the candidates again, to pick out parse and violation profile.
        # Each element of candidates is a (<candidate>, <violation profile>) tuple.
        candidates_match = re.findall(candidate_pattern, t)
        
        # Following for-loop is identical to overt_tableaux
        overt_evals = {}
        for match in candidates_match:
            # The candidate string for an RIP includes both the parse and the output.
            # I.e., "/output/ \-> [parse]"
            parse_and_overt = match[0]
            if len(re.findall(rip_pattern, parse_and_overt)) != 1:
                raise ValueError("Candidate "+match[0]+" doesn't look like an RIP candidate. Please check grammar file.")
            overt = re.findall(rip_pattern, parse_and_overt)[0][0]
            parse = re.findall(rip_pattern, parse_and_overt)[0][1]
            viols_string = match[1]

            viols = viols_string.rstrip().split(' ')
            viols = [int(x) for x in viols] 

            viol_profile = map_lists_to_dict(consts, viols)
            overt_evals[(overt, parse)] = viol_profile
            
        tableaux[inp] = overt_evals
    
    return tableaux

def build_tableaux_RIP_i2p(grammar_string):
    tableaux_string = re.findall(tableau_pattern, grammar_string)
    tableaux_string = [t[0] for t in tableaux_string]
    consts = re.findall(const_pattern, grammar_string)
    consts = [c[0] for c in consts]
    
    tableaux = {}
    for t in tableaux_string:
        # Since there's only one input form per tableau,
        # re.findall should always yield a list of length 1)
        if len(re.findall(input_pattern, t)) == 0:
            raise ValueError("No input found in the following tableaux_string. Pleae check grammar file.\n"+t)
        elif len(re.findall(input_pattern, t)) > 1:
            raise ValueError("Found more than one input form in tableau. Please check grammar file.")

        inp = re.findall(input_pattern, t)[0]

        # Access the candidates again, to pick out parse and violation profile.
        # Each element of candidates is a (<candidate>, <violation profile>) tuple.
        candidates_match = re.findall(candidate_pattern, t)
        
        # Following for-loop is identical to overt_tableaux
        parse_evals = {}
        for match in candidates_match:
            # The candidate string for an RIP includes both the parse and the output.
            # I.e., "/output/ \-> [parse]"
            parse_and_overt = match[0]
            if len(re.findall(rip_pattern, parse_and_overt)) != 1:
                raise ValueError("Candidate "+match[0]+" doesn't look like an RIP candidate. Please check grammar file.")
            overt = re.findall(rip_pattern, parse_and_overt)[0][0]
            parse = re.findall(rip_pattern, parse_and_overt)[0][1]
            viols_string = match[1]

            viols = viols_string.rstrip().split(' ')
            viols = [int(x) for x in viols] 

            viol_profile = map_lists_to_dict(consts, viols)
            parse_evals[parse] = viol_profile
            
        tableaux[inp] = parse_evals
    
    return tableaux

# Only RIP needs to build overt tableaux
def build_tableaux_RIP_o2p(grammar_string):
    tableaux_string = re.findall(tableau_pattern, grammar_string)
    tableaux_string = [t[0] for t in tableaux_string]
    consts = re.findall(const_pattern, grammar_string)
    consts = [c[0] for c in consts]
    
    overt_tableaux = {}
    for t in tableaux_string:
        # Since the parentheses in the candidate_pattern regex capture these three string groups,
        # re.findall returns the list of (<overt form>, <parse>, <violation profile>) tuples.
        candidates_match = re.findall(candidate_pattern, t)

        overt_set = []
        candidates = []
        # A match is a (<overt form>, <parse>, <violation profile>) tuple
        for match in candidates_match: 
            parse_and_overt = match[0]
            if len(re.findall(rip_pattern, parse_and_overt)) != 1:
                raise ValueError("Candidate "+cand+" doesn't look like an RIP candidate. Please check grammar file.")
            overt = re.findall(rip_pattern, parse_and_overt)[0][0]
            parse = re.findall(rip_pattern, parse_and_overt)[0][1]
            viols_string = match[1]
            overt_set.append(overt)
            candidates.append((overt, parse, viols_string))
        # Remove duplicates from overt_set
        overt_set = set(overt_set)

        # Each overt form will be a key of overt_tableaux.
        # The value of each overt form will be a parse_evals dictionary.
        for overt in overt_set:
            # The keys of a parse_evals are the parses of the affiliated overt form.
            # The value of a parse is its violation profile.
            parse_evals = {}

            for cand in candidates:
                cand_overt = cand[0]
                cand_parse = cand[1]
                cand_viols_string = cand[2]
                

                # Pick out the cand tuples affiliated with the overt form.
                if cand_overt == overt:
                    # convert violation profile from string to list
                    # E.g., from '0 1 0' to ['0', '1', '0']
                    viols = cand_viols_string.rstrip().split(' ')
                    # convert string (e.g., '0') to integer (e.g., 0)
                    viols = [int(x) for x in viols] 

                    # Map the list of constraints with list of violations,
                    # so that the value of the dictionary is ((CONST_NAME, VIOL), (CONST_NAME, VIOL), ...)
                    viol_profile = map_lists_to_dict(consts, viols)
                    
                    parse_evals[cand_parse] = viol_profile

            overt_tableaux[overt] = parse_evals

    return overt_tableaux

# Make constraint dictionary
def const_dict(grammar_string, initiate=True, init_value=None):
    const_dict = {}
    consts_rv = re.findall(const_pattern, grammar_string)
    if initiate:
        for const in consts_rv:
            const_dict[str(const[0])] = float(init_value)
    else:
        for const in consts_rv:
            const_dict[str(const[0])] = float(const[1])
    return const_dict

### With a tableau and a constraint dictionary, we can build "grammar" objects
# Non-RIP GLA takes as its argument an object of class "grammar".
class grammar:
    def __init__(self, grammar_string):
        self.i2o_tableaux = build_tableaux(grammar_string)
        self.const_dict = const_dict(grammar_string, initiate=False)

# RIP/GLA takes as its argument an object of class "grammar_RIP".
class grammar_RIP:
    def __init__(self, grammar_string):
        self.i2p_tableaux = build_tableaux_RIP_i2p(grammar_string)
        self.o2p_tableaux = build_tableaux_RIP_o2p(grammar_string)
        self.i2o_tableaux = build_tableaux_RIP_i2o(grammar_string)
        self.const_dict = const_dict(grammar_string, initiate=False)

# These init grammars are for building a templatic grammar where the
# ranking values of all constraints are equally assigned to some default value.
# (Default set to 100 if not specified otherwise)
class grammar_init:
    def __init__(self, grammar_string, init_value=100):
        self.i2o_tableaux = build_tableaux(grammar_string)
        self.const_dict = const_dict(grammar_string, True, init_value)

class grammar_init_RIP:
    def __init__(self, grammar_string, init_value=100):
        self.i2p_tableaux = build_tableaux_RIP_i2p(grammar_string)
        self.o2p_tableaux = build_tableaux_RIP_o2p(grammar_string)
        self.i2o_tableaux = build_tableaux_RIP_i2o(grammar_string)
        self.const_dict = const_dict(grammar_string, True, init_value)

################################################################################
##### Part 2: Defining utility functions #######################################
################################################################################

### Used for non-RIP/GLA. Find an input, given an output.
def find_input(overt_string, input_tableaux):
    potential_inps = []
    for inp in input_tableaux.keys():
        if overt_string in input_tableaux[inp].keys():
            potential_inps.append(inp)
    if len(potential_inps) == 0:
        raise ValueError("No input found: "+overt_string+" is not a candidate in this grammar file.")
    return potential_inps

### Output is not 'found' from the tableaux in RIP-GLA.
# It is reverse-engineered by stripping a word of its stress pattern.
def make_input(overt_string):
    core_pattern = re.compile(r"\[(.*)\]")
    if not re.search(core_pattern, overt_string):
        raise ValueError("Format of overt form "+overt_string+" is not appropriate. It should look like '[L1 H H]'.")

    core = re.search(core_pattern, overt_string).group(1)
    core = re.sub(r"\d", "", core)
    inp = "|"+core+"|"
    return inp

### Add noise to ranking values of each constraint (for noisy learning)
# Noise follows gaussian dist of mean 0, and specified sigma value (default 2).
def add_noise(const_dict, noise_sigma=2.0):
    const_dict_copy = const_dict.copy()
    for const in const_dict.keys():
        noise = random.gauss(0, noise_sigma)
        const_dict_copy[const] = const_dict[const] + noise
    return const_dict_copy

# Rank constraints in const_dict by their ranking value and return an ordered list
def ranking(const_dict):
    ranked_list_raw=[]
    for const in const_dict:
        ranked_list_raw.append((const, const_dict[const]))
    # Random shuffle raw list to get rid of the effects of Python's default ordering
    random.shuffle(ranked_list_raw) 
    ranked_list_raw = sorted(ranked_list_raw, key=itemgetter(1, 0), reverse=True)
    ranked_list = [x[0] for x in ranked_list_raw]
    return ranked_list

### A recursive function that does run-of-the-mill OT
# It takes as argument a special dictionary, tableau_viol_only, which acts as a 
# sub-tableau of sorts. The user never needs to manually provide it, since the 
# function is embedded in other functions.
def optimize(tableau_viol_only):
    # Pick out the most serious offense of each parse
    # (I.e., pick out the highest-ranked constraint violated by the parse)
    initial_batch = []
    for value in tableau_viol_only.values():
        # The value is a list of (parse, const_rank, const, viol) tuples, sorted by const_rank.
        # The first element of this list is the "most serious offense."
        initial_batch.append(value[0])

    # Among the most serious offense commited by each parse, 
    # pick out the parse(s) that committed the least serious one.
    lowest_rank_compare = []
    # max, because the *largest* const_rank value means least serious
    lowest_rank = max(initial_batch, key = lambda x:x[1])
    for parse in initial_batch:
        if parse[1] == lowest_rank[1]:
            lowest_rank_compare.append(parse)

    # If there is a single parse with the least serious offense, that's the winner.
    if len(lowest_rank_compare) == 1:
        return lowest_rank_compare[0]

    # If there are more than one least-serious offenders...
    elif len(lowest_rank_compare) > 1:
        # ... we first see whether one has violated the same constraint more than the other(s).
        viol_compare = []
        lowest_viol = min(lowest_rank_compare, key = lambda x:x[3])
        for x in lowest_rank_compare:
            if x[3] == lowest_viol[3]:
                viol_compare.append(x)

        # If there is one parse that violated the constraint the least, that's the winner.
        if len(viol_compare) == 1:
            return viol_compare[0]
        
        # If all of the least-serious offenders violated the constraint the same number of times,
        # we now need to compare their next most serious constraint offended.
        elif len(viol_compare) > 1:
            partial_tableau_viol_only = {}
            for x in viol_compare:
                # Make another tableau_viol_only with the least-serious offenders,
                # but we chuck out their most serious offenses.
                partial_tableau_viol_only[x[0]] = tableau_viol_only[x[0]][1:]
            # Run the algorithm again with the new, partial tableau_viol_only
            return optimize(partial_tableau_viol_only)
        else:
            raise ValueError("Could not find optimal candidate")
    else:
        raise ValueError("Could not find optimal candidate")

### Function that creates the tableaux_viol_only argument for above function

def tableaux_viol_only(inp, ranked_consts, tableaux):
    # Pick out the constraints that *are* violated (i.e., violation > 0)
    # This "sub-dictionary" will be fed into the optimize function
    tableau_viol_only = {}
    for parse in tableaux[inp].keys():
        tableau_viol_only[parse] = []
        for const, viol in tableaux[inp][parse].items():
            if viol > 0:
                tableau_viol_only[parse].append((parse, ranked_consts.index(const), const, viol))
        tableau_viol_only[parse] = sorted(tableau_viol_only[parse], key = lambda x:x[1])
    
    return tableau_viol_only

### This is the wrapper function for "optimize"
# Given an input, a grammar, and violation profiles, it performs the "optimize"
# function and yields the tuple of (winner, winner's_viol_profile)
def generate(inp, ranked_consts, tableaux):
    # Pick out the constraints that *are* violated (i.e., violation > 0)
    # This "sub-dictionary" will be fed into the optimize function

    sub_tableau = tableaux_viol_only(inp, ranked_consts, tableaux)

    gen_parse = optimize(sub_tableau)[0]
    gen_viol_profile = tableaux[inp][gen_parse]
    
    return (gen_parse, gen_viol_profile)


    

# Adjusting the grammar, given the list of good and bad constraints
def adjust_grammar(good_consts, bad_consts, const_dict, plasticity=1.0):
    for const in good_consts:
        const_dict[const] = const_dict[const] + float(plasticity/len(good_consts))
    for const in bad_consts:
        const_dict[const] = const_dict[const] - float(plasticity)
    return const_dict

# In the face of an error, classify constraints into good, bad, and irrelevant constraints.
def learn(winner_viol_profile, loser_viol_profile, const_dict, plasticity):
    good_consts = [] # Ones that are violated more by the "wrong" parse than by the actual datum
    bad_consts = [] # Ones that are violated more by actual datum than by the "wrong" parse
    for const in winner_viol_profile.keys():
        if winner_viol_profile[const] > loser_viol_profile[const]:
            bad_consts.append(const)
        elif winner_viol_profile[const] < loser_viol_profile[const]:
            good_consts.append(const)
        else: # equal number of violations for the parse and the datum
            continue
    # Adjust the grammar according to the contraint classifications
    return adjust_grammar(good_consts, bad_consts, const_dict, plasticity)


################################################################################
##### Part 3: The actual learning ##############################################
################################################################################

### Given a list of observed words in the language and a grammar object,
# figure out the ranking of the constraints of the grammar object that is 
# compatible with all of the words.

### Learning function for non-RIP/GLA
def do_learning(target_list, grammar, plasticity=1.0, noise_bool=True, noise_sigma=2.0, print_bool=True, print_cycle=1000):
    i2o_tableaux = grammar.i2o_tableaux
    const_dict = grammar.const_dict
    
    target_list_shuffled = random.sample(target_list, len(target_list))
    target_set = set(target_list)

    datum_counter = 0
    change_counter = 0
    learned_list = []

    # Data to be plotted
    # track the iteration number where change occurred
    # (will plot the interval between changes)
    interval_track = [] 
    # track number of learned tokens
    learning_track = []
    # Track ranking values for each constraint
    ranking_value_tracks = {}
    for const in const_dict.keys():
        ranking_value_tracks[const] = []

    for t in target_list_shuffled:
        datum_counter += 1

        inp = find_input(t, i2o_tableaux)[0]
        if noise_bool==True:    
            generation = generate(inp, ranking(add_noise(const_dict, noise_sigma)), i2o_tableaux)
        else:
            generation = generate(inp, ranking(const_dict), i2o_tableaux)

        if generation[0] == t:
            learned_list.append(t)

            ### Export information for plotting
            for const in ranking_value_tracks.keys():
                ranking_value_tracks[const].append(const_dict[const])
        else:
            change_counter += 1
            # new grammar
            const_dict = learn(i2o_tableaux[inp][t], generation[1], const_dict, plasticity)
            # new generation with new grammar
            generation = generate(inp, ranking(const_dict), i2o_tableaux)

            ### Export information for plotting
            for const in ranking_value_tracks.keys():
                ranking_value_tracks[const].append(const_dict[const])
            
            interval_track.append(datum_counter)
        
        ### Export information for plotting
        learning_track.append(len(learned_list))

        if print_bool==True and datum_counter % print_cycle == 0:
            print(str(datum_counter)+" out of "+str(len(target_list_shuffled))+" learned")
    
    learned_set = set(learned_list)
    failed_set = target_set.difference(learned_set)

    return (const_dict, change_counter, len(target_list), failed_set, plasticity, noise_bool, noise_sigma, ranking_value_tracks, learning_track, interval_track)


### A "learning" class object which simply wraps the do_learning function
# and makes its resulting object callable with keywords.
class learning:
    def __init__(self, target_list, grammar, plasticity=1.0, noise_bool=True, noise_sigma=2.0, print_bool=True, print_cycle=1000):
        results = do_learning(target_list, grammar, plasticity, noise_bool, noise_sigma, print_bool, print_cycle)
        self.const_dict = results[0]
        self.change_counter = results[1]
        self.num_of_data = results[2]
        self.failed_set = results[3]
        self.plasticity = results[4]
        self.noise_bool = results[5]
        self.noise_sigma = results[6]
        self.ranking_value_tracks = results[7]
        self.learning_track = results[8]
        self.interval_track = results[9]
        self.grammar = grammar
        self.target_list = target_list


### Similar to the do_learning function, but for RIP/GLA.
def do_learning_RIP(target_list, grammar_RIP, plasticity=1.0, noise_bool=True, noise_sigma=2.0, print_bool=True, print_cycle=1000):

    #logfilename = timestamp_filepath('txt', 'log')
    #logfile = open(logfilename, 'w')

    i2p_tableaux = grammar_RIP.i2p_tableaux
    o2p_tableaux = grammar_RIP.o2p_tableaux
    i2o_tableaux = grammar_RIP.i2o_tableaux
    const_dict = grammar_RIP.const_dict
    
    target_list_shuffled = random.sample(target_list, len(target_list))
    target_set = set(target_list)

    datum_counter = 0
    change_counter = 0
    learned_list = []

    # Data to be plotted
    # track the iteration number where change occurred
    # (will plot the interval between changes)
    interval_track = [] 
    # track number of learned tokens
    learning_track = []
    # Track ranking values for each constraint
    ranking_value_tracks = {}
    for const in const_dict.keys():
        ranking_value_tracks[const] = []


    for t in target_list_shuffled:
        datum_counter += 1

        if noise_bool==True:
            const_dict_noisy = add_noise(const_dict, noise_sigma)
            ranked_consts = ranking(const_dict_noisy)
        elif noise_bool==False:
            ranked_consts = ranking(const_dict)
        else:
            raise ValueError("Please verify value of noise_bool")

        generation = generate(make_input(t), ranked_consts, i2p_tableaux)
        rip_parse = generate(t, ranked_consts, o2p_tableaux)

        if generation[0] == rip_parse[0]:
            learned_list.append(t)

            ### Export information for plotting
            for const in ranking_value_tracks.keys():
                ranking_value_tracks[const].append(const_dict[const])

        else:
            change_counter += 1
            # new grammar
            const_dict = learn(rip_parse[1], generation[1], const_dict, plasticity)
            ranked_consts = ranking(const_dict)
            # new generation with new grammar
            generation = generate(make_input(t), ranked_consts, i2p_tableaux)
            # new rip parse with new grammar
            rip_parse = generate(t, ranked_consts, o2p_tableaux)

            ### Export information for plotting
            for const in ranking_value_tracks.keys():
                ranking_value_tracks[const].append(const_dict[const])
            
            interval_track.append(datum_counter)
        
        ### Export information for plotting
        learning_track.append(len(learned_list))

        if print_bool and datum_counter % print_cycle == 0:
            print(str(datum_counter)+" out of "+str(len(target_list_shuffled))+" learned")

    learned_set = set(learned_list)
    failed_set = target_set.difference(learned_set)

    #logfile.close()

    return (const_dict, change_counter, len(target_list), failed_set, plasticity, noise_bool, noise_sigma, ranking_value_tracks, learning_track, interval_track)

### Similarly, a learning_RIP class object to make the results callable.
class learning_RIP:
    def __init__(self, target_list, grammar_RIP, plasticity=1.0, noise_bool=True, noise_sigma=2.0, print_bool=True, print_cycle=1000):
        results = do_learning_RIP(target_list, grammar_RIP, plasticity, noise_bool, noise_sigma, print_bool, print_cycle)
        self.const_dict = results[0]
        self.change_counter = results[1]
        self.num_of_data = results[2]
        self.failed_set = results[3]
        self.plasticity = results[4]
        self.noise_bool = results[5]
        self.noise_sigma = results[6]
        self.ranking_value_tracks = results[7]
        self.learning_track = results[8]
        self.interval_track = results[9]
        self.grammar = grammar_RIP
        self.target_list = target_list
        self.learning_type = 'og_RIP'


################################################################################
##### Part 4: ERC-based target choice ##########################################
################################################################################

'''
This part is a amendment to the original configuration of the GLA, 
which I propose in my 2nd generals paper. Instead of choosing the most optimal 
candidate under the current (faulty) OT grammar, the learner chooses the
candidate which involves the least amount of change to the current grammar.
(I.e., the candidate for which the distance between the undominated L and the
highest W is the shortest. See Prince (2002) on LWe notation, known as the ERC.)
If there are multiple such candidates, choose randomly among them.
'''

### Take as argument the dictionary returned by rip_LW_pairs,
# identify the undominated L (if it exists) and the highest W,
# and return another dictionary with each parse as the key and
# its LW distance as value.
def LW_distance_picker(LW_pairs_dict):
    LW_dists = {}
    for parse in LW_pairs_dict.keys():
        Ls = []
        es = []
        Ws = []
        for tup in LW_pairs_dict[parse]:
            if tup[1] == 'e':
                es.append(tup)
            elif tup[1] == 'L':
                Ls.append(tup)
            elif tup[1] == 'W':
                Ws.append(tup)

        gen_parse = LW_pairs_dict[parse][0][3]

        if len(Ls)+len(es)+len(Ws) != 12:
            raise ValueError("The LWes for "+parse+" against "+gen_parse+" not completed correctly. Num of W, L, e does not add up to 12.")
        elif len(Ls) == 0:
            raise ValueError("The LWes for "+parse+" against "+gen_parse+" not completed correctly. Num of Ls is 0.")
      
        worst_L = min(Ls, key=lambda x:x[2])
        # The list of Ws dominating the highest L
        dominating_Ws = [W for W in Ws if W[2]<worst_L[2]]
        dominated_Ws = [W for W in Ws if W[2]>worst_L[2]]

        if len(dominating_Ws) > 0:
            LW_dists[parse] = 0
        elif len(Ws) == 0:
            LW_dists[parse] = 20
        else:
            problem_W = min(dominated_Ws, key=lambda x:x[2])
            distance = int(problem_W[2] - worst_L[2])
            LW_dists[parse] = distance

    return LW_dists

### Given an overt form, ranked constraints, and an o2p tableau,
# return a dictionary whose keys are all parses compatible with the overt form.
# The value for each parse is a list of tuples. Each tuple contains a constraint,
# and whether the constraint is an L, W, or e against the optimal parse of the overt form.
def rip_LW_pairs(overt, gen_viol_profile, gen_parse, ranked_consts, i2o_tableaux, o2p_tableaux):  
    LW_pairs = {}
    for parse in o2p_tableaux[overt].keys(): 
        LW_pairs[parse] = []

        for const in o2p_tableaux[overt][parse].keys():
            # Winner (RIP-parse) preferrer
            if gen_viol_profile[const] > o2p_tableaux[overt][parse][const]:
                LW_pairs[parse].append((const, 'W', ranked_consts.index(const), gen_parse))
            # Loser (gen_parse) preferrer
            elif gen_viol_profile[const] < o2p_tableaux[overt][parse][const]:
                LW_pairs[parse].append((const, 'L', ranked_consts.index(const), gen_parse))
            # Equal
            elif gen_viol_profile[const] == o2p_tableaux[overt][parse][const]:
                LW_pairs[parse].append((const, 'e', ranked_consts.index(const), gen_parse))
            # Raise error if none of the above
            else:
                raise ValueError("Failed to compare LW profile for parse "+parse)
    return LW_pairs


# Function to do a revised form of RIP learning using LW distance
def do_revised_learning_RIP(target_list, grammar_RIP, plasticity=1.0, noise_bool=True, noise_sigma=2.0, print_bool=True, print_cycle=1000):

    #logfilename = timestamp_filepath('txt', 'log')
    #logfile = open(logfilename, 'w')

    i2p_tableaux = grammar_RIP.i2p_tableaux
    o2p_tableaux = grammar_RIP.o2p_tableaux
    i2o_tableaux = grammar_RIP.i2o_tableaux
    const_dict = grammar_RIP.const_dict
    
    target_list_shuffled = random.sample(target_list, len(target_list))
    target_set = set(target_list)

    datum_counter = 0
    change_counter = 0
    learned_list = []

    # Data to be plotted
    # track the iteration number where change occurred
    # (will plot the interval between changes)
    interval_track = [] 
    # track number of learned tokens
    learning_track = []
    # Track ranking values for each constraint
    ranking_value_tracks = {}
    for const in const_dict.keys():
        ranking_value_tracks[const] = []


    for t in target_list_shuffled:
        datum_counter += 1

        if noise_bool==True:
            const_dict_noisy = add_noise(const_dict, noise_sigma)
            ranked_consts = ranking(const_dict_noisy)
        elif noise_bool==False:
            ranked_consts = ranking(const_dict)
        else:
            raise ValueError("Please verify value of noise_bool")

        generation = generate(make_input(t), ranked_consts, i2p_tableaux)
        rip_parse = generate(t, ranked_consts, o2p_tableaux)

        if generation[0] == rip_parse[0]:
            learned_list.append(t)

            ### Export information for plotting
            for const in ranking_value_tracks.keys():
                ranking_value_tracks[const].append(const_dict[const])

        else:
            change_counter += 1

            # find the optimal rip parse            
            LW_pairs = rip_LW_pairs(t, generation[1], generation[0], ranking(const_dict), i2o_tableaux, o2p_tableaux)
            distances = LW_distance_picker(LW_pairs)

            mindist = min(distances.values())
            mindist_candidates = [x for x in distances.keys() if distances[x] == mindist]

            if len(mindist_candidates) == 1:
                target_parse = mindist_candidates[0]
            elif len(mindist_candidates) > 1:
                target_parse = random.sample(mindist_candidates, 1)[0]
            
            target_viol_profile = o2p_tableaux[t][target_parse]
            
            # new grammar
            const_dict = learn(target_viol_profile, generation[1], const_dict, plasticity)
            ranked_consts = ranking(const_dict)
            # new generation with new grammar
            #generation = generate(make_input(t), ranked_consts, i2p_tableaux)
            # new rip parse with new grammar
            #rip_parse = generate(t, ranked_consts, o2p_tableaux)

            ### Export information for plotting
            for const in ranking_value_tracks.keys():
                ranking_value_tracks[const].append(const_dict[const])
            
            interval_track.append(datum_counter)
        
        ### Export information for plotting
        learning_track.append(len(learned_list))

        if print_bool and datum_counter % print_cycle == 0:
            print(str(datum_counter)+" out of "+str(len(target_list_shuffled))+" learned")

    learned_set = set(learned_list)
    failed_set = target_set.difference(learned_set)

    #logfile.close()

    return (const_dict, change_counter, len(target_list), failed_set, plasticity, noise_bool, noise_sigma, ranking_value_tracks, learning_track, interval_track)

# Class that acts as wrapper of the function do_revised_learning_RIP
class revised_learning_RIP:
    def __init__(self, target_list, grammar_RIP, plasticity=1.0, noise_bool=True, noise_sigma=2.0, print_bool=True, print_cycle=1000):
        results = do_revised_learning_RIP(target_list, grammar_RIP, plasticity, noise_bool, noise_sigma, print_bool, print_cycle)
        self.const_dict = results[0]
        self.change_counter = results[1]
        self.num_of_data = results[2]
        self.failed_set = results[3]
        self.plasticity = results[4]
        self.noise_bool = results[5]
        self.noise_sigma = results[6]
        self.ranking_value_tracks = results[7]
        self.learning_track = results[8]
        self.interval_track = results[9]
        self.grammar = grammar_RIP
        self.target_list = target_list
        self.learning_type = 'LW_RIP'


################################################################################
##### Part 5: Output Results ######## ##########################################
################################################################################

# Timestamp for filenames
def timestamp_filepath(extension, label=''):
    # Timestamp for file
    yy = str(datetime.datetime.now())[2:4]
    mm = str(datetime.datetime.now())[5:7]
    dd = str(datetime.datetime.now())[8:10]
    hh = str(datetime.datetime.now())[11:13]
    mn = str(datetime.datetime.now())[14:16]
    ss = str(datetime.datetime.now())[17:19]
    timestamp = yy+mm+dd+"_"+hh+mn+ss

    # Designate absolute path of results file and open it
    script_path = os.path.dirname(os.path.realpath(sys.argv[0])) #<-- absolute dir the script is in
    if label != '':
        output_path = script_path + '\\results\\'+label
    else:
        output_path = script_path + '\\results'
    output_file_name = "\\"+label+"_"+timestamp+'.'+extension
    output_file_path = output_path + output_file_name

    return output_file_path

### Evaluating function
# At the end of a trial, produce the output for each input according
# to the final ranking constraint obtained by learning
# Compare the output with the actual word of the language,
# and flag if the two are different.

# This is eval for non-RIP GLA
def eval_errors(learning, num):
    target_list = learning.target_list
    ranked_consts = ranking(learning.const_dict)
    used_grammar = learning.grammar
    tableaux = used_grammar.i2o_tableaux
    error_list = []

    i = 0
    while i < num:
        i += 1
        t = random.sample(target_list, 1)[0]
        learned_form = generate(find_input(t, tableaux)[0], ranked_consts, tableaux)[0]
        if learned_form != t:
            print("Eval error: Learned "+learned_form+', target '+t)
            error_compare = ' '.join([t, learned_form])
            error_list.append(error_compare)
    return error_list

# Noisy evaluation for RIP GLA
def eval_errors_RIP_noisy(learning):
    target_list = learning.target_list
    used_grammar = learning.grammar
    i2o_tableaux = used_grammar.i2o_tableaux
    o2p_tableaux = used_grammar.o2p_tableaux
    error_list = []
    error_count = {}
    scrambled_target_list = random.sample(target_list, len(target_list))

    for t in scrambled_target_list:
        const_dict = learning.const_dict
        ranked_consts = ranking(add_noise(const_dict, 2.0))
        learned_form = generate(make_input(t), ranked_consts, i2o_tableaux)[0][0]
        if learned_form != t:
            learned_parse = generate(learned_form, ranked_consts, o2p_tableaux)[0]
            RIP_parse = generate(t, ranked_consts, o2p_tableaux)[0]
            error_profile = (t, RIP_parse, learned_form, learned_parse)
            if error_profile in error_count.keys():
                error_count[error_profile] += 1
            else:
                error_count[error_profile] = 1
    for error_profile in error_count.keys():
        error_profile_count = list(error_profile) + [str(error_count[error_profile])]
        error_compare = ' '.join(error_profile_count)
        error_list.append(error_compare)
    return error_list

# Non-noisy evaluation for RIP GLA
def eval_errors_RIP_nonnoisy(learning):
    target_set = set(learning.target_list)
    used_grammar = learning.grammar
    ranked_consts = ranking(learning.const_dict)
    i2o_tableaux = used_grammar.i2o_tableaux
    o2p_tableaux = used_grammar.o2p_tableaux
    error_list = []

    for t in target_set:
        learned_form = generate(make_input(t), ranked_consts, i2o_tableaux)[0][0]
        if learned_form != t:
            learned_parse = generate(learned_form, ranked_consts, o2p_tableaux)[0]
            error_compare = ' '.join([t, learned_form, learned_parse])
            error_list.append(error_compare)
    return error_list

# Plot results
def plot_results(learning_result, plot_rvs=True, plot_learning=True, plot_intervals=True, save=True, label=''):
    num_of_data = learning_result.num_of_data
    iteration_track = list(range(1, num_of_data+1))
    ranking_value_tracks = learning_result.ranking_value_tracks
    learning_track = learning_result.learning_track
    interval_track = learning_result.interval_track

    list_of_plots = []
    if plot_rvs == True:
        list_of_plots.append('rvs')
    
    if plot_learning == True:
        list_of_plots.append('learning')
    
    if plot_intervals == True:
        list_of_plots.append('intervals')
    
    if len(list_of_plots) == 0:
        raise ValueError("None of the plot parameters were turned on ('True').")

    plt.figure()

    for p in list_of_plots:
        if p == 'rvs':
            plt.subplot(len(list_of_plots), 1, list_of_plots.index(p)+1)
            for const in ranking_value_tracks.keys():
                plt.plot(iteration_track, ranking_value_tracks[const], label=const)
                plt.text(55000, ranking_value_tracks[const][-1], const, fontsize=8)
        elif p == 'learning':
            plt.subplot(len(list_of_plots), 1, list_of_plots.index(p)+1)
            plt.plot(iteration_track, learning_track)
        elif p == 'intervals':
            intervals = []
            changes = []
            for i in range(0, len(interval_track)-1):
                intervals.append(interval_track[i+1]-interval_track[i])
                changes.append(i+1)
            plt.subplot(len(list_of_plots), 1, list_of_plots.index(p)+1)
            plt.plot(changes, intervals)
    
    if save==True:
        figure_file_path = timestamp_filepath('svg', label)
        plt.savefig(figure_file_path)
        print("Figure file: "+figure_file_path)
    else:
        plt.show()

# Write results to a txt file
def write_results(learning_result, label='', is_RIP=None, noisy_eval=True):
    const_dict = learning_result.const_dict
    change_counter = learning_result.change_counter
    num_of_data = learning_result.num_of_data
    failed_set = learning_result.failed_set
    plasticity = learning_result.plasticity
    noise_bool = learning_result.noise_bool
    noise_sigma = learning_result.noise_sigma
    interval_track = learning_result.interval_track
    
    results_file_path = timestamp_filepath('txt', label)
    results_file = open(results_file_path, 'w')

    # Write title
    if is_RIP == True:
        results_file.write("RIP/OT-GLA learning results\n")
    elif is_RIP == False:
        results_file.write("OT-GLA learning results\n")
    else:
        results_file.write("OT-GLA learning results (unknown if RIP or not)\n")
    
    
    # Write "learning type" (OG RIP? amended LW RIP?)
    if learning_result.learning_type == 'og_RIP':
        results_file.write("Algorithm: Original Configuration\n")
    elif learning_result.learning_type == 'LW_RIP':
        results_file.write("Algorithm: Amended(LW) Configuration\n")

    # Write how many times the grammar was changed
    results_file.write("Grammar changed "+str(change_counter)+"/"+str(num_of_data)+" times\n")

    # Write plasticity and noise settings
    results_file.write("Plasticity: "+str(plasticity)+"\n")
    if noise_bool == True:
        results_file.write("Noise: "+str(noise_sigma)+"\n")
    else:
        results_file.write("Noise: No noise\n\n")

    # Write actual ranking values
    results_file.write("Constraints and ranking values\n")
    for const in ranking(const_dict):
        results_file.write(const+"\t"+str(const_dict[const])+"\n")

    # If there were any datum types never learned, print them
    if len(failed_set) > 0:
        results_file.write("Overt forms that were never learned:")
        for i in failed_set:
            results_file.write(str(i)+"\n")
    
    if is_RIP == True:
        if noisy_eval == True:
            errors = eval_errors_RIP_noisy(learning_result)
        elif noisy_eval == False:
            errors = eval_errors_RIP_nonnoisy(learning_result)
    else:
        errors = eval_errors(learning_result)
    
    if len(errors) == 0:
        results_file.write("\nNo errors found in evaluation")
    elif len(errors) > 0:
        results_file.write("\n"+str(len(errors))+" words not (fully) learned in evaluation (target, learned form, (learned parse), count):\n")
        for e in errors:
            results_file.write(e+"\n")

    results_file.close()
    print("Output file: "+results_file_path)    



if __name__ == "__main__":
    pass
