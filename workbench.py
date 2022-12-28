from gla import *
import random
import sys
import logging
import logging.handlers

# Print out errors to log file
#logging.basicConfig(filename='workbenchlog.log',
#                    format='%(asctime)s %(message)s')

try:
    # Initiate a grammar_RIP object from the Praat otgrammar template
    template_grammar_string = grammar_string('./grammars/PraatMetricalGrammar_2stress.txt')
    template_grammar = grammar_init_RIP(template_grammar_string, init_value=100)

    # Create the three tableaux for RIP learning
    template_i2o_tableaux = template_grammar.i2o_tableaux
    template_i2p_tableaux = template_grammar.i2p_tableaux
    template_o2p_tableaux = template_grammar.o2p_tableaux

    if int(sys.argv[1]) < 10:
        target_num = '0'+str(sys.argv[1])
    else:
        target_num = str(sys.argv[1])

    target_filename = './languages/hypo'+target_num+'_data.txt'

    target_types = read_and_rstrip(target_filename)
    target_tokens = []
    for i in range(1000):
        for t in target_types:
            target_tokens.append(t)

    targets = random.sample(target_tokens, len(target_tokens))

    learning = revised_learning_RIP(targets, template_grammar, print_cycle=1000)
    #learning = learning_RIP(targets, template_grammar, print_cycle=1000)

    result_label = 'hypo'+target_num

    write_results(learning, result_label, is_RIP=True, noisy_eval=False)

    #plot_results(learning, True, False, True, True, result_label)
except IndexError:
    # When running this code, please choose a language number to learn
    print("Error: Please specify the language number to be learned by the GLA.")
else:
    # Log all other errors to log file.
    logging.exception("message")


