## Update on December 28, 2022 (reflected in Proceedings of AMP 2022)

**1. Removed two languages (original Lang 1 and Lang 2) because there was overlap.**  

After presentation of the study at AMP 2022 in October 2022, I realized that two of the 68 languages initially reported (Lang 1 and Lang 2) were not the ones randomly sampled with Praat, but ones I created separately at an earlier stage of the project and mistakenly mixed into the bunch. Since they happened to be identical with Lang 14 and Lang 8 respectively, I excluded Lang 1 and Lang 2 and recalculated the mean performance and mean difference in number of grammar changes for a total of 66 languages.

**2. Re-labeled Lang 67 and Lang 68 as the new "Lang 1" and "Lang 2" respectively.**  

I also re-labeled Lang 67 and Lang 68, the last two languages, as the new Lang 1 and 2 in the .xlsx and .csv files in the [results folder and its subdirectories](results/). This makes the very last label in the word list "Lang 66", hence avoiding confusion about the total number of languages -- especially in the proceedings paper. (Results from original Lang 1 and original Lang 2 are still preserved, but the directories are clearly labeled "EXCLUDED".)

***This update does not affect the crucial findings about Lang 55 and lang 3, nor does it change the general conclusion of this study***. But I believe it is a more accurate reflection of the nature of the project.

The older repository is preserved here: https://github.com/EunsunJou/Economy_RIPGLA_Old

# Economy_RIPGLA

This repository contains code and results files from my work on economy-based amendment of Robust Interpretive Parsing.
Specifically, it contains:

1. A python implementation of the RIP/OT-GLA, available as the module gla.py
2. An economy-based amendment to the algorithm (the RIP/ERC-GLA), also available in the gla.py module
3. A set of 66 randomly sampled abstract metrical stress languages, available in [languages](/languages)  
(These are called Lang 1, Lang 2, ... in the paper but the files and folders say "hypo01", "hypo02", ...)
4. Files resulting from learning trials using the RIP/GLA and its amended version, available in [results](/results)

gla.py is intended to be loaded as a module in a separate python script.
workbench.py is an example script that loads the module and actually does the learning.
