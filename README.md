## Update on December 28, 2022 (reflected in Proceedings of AMP 2022)

1. Removed two languages (original Lang 1 and Lang 2) because there was overlap.
2. Renamed Lang 67 and Lang 68 as "Lang 1" and "Lang 2" respectively, so that the very last label would be Lang 66. This avoids confusion about total number of languages.

After presentation of the study at AMP 2022 in October 2022, I realized that two of the 68 languages initially reported (labeled Lang 1 and Lang 2) were not the ones randomly sampled with Praat, but ones I created separately at an earlier stage of the project. Since they happened to be identical with Lang 14 and Lang 8 respectively, I excluded Lang 1 and Lang 2 and recalculated the mean performance and mean difference in number of grammar changes for **66 languages**. I also renamed Lang 67 and Lang 68 to Lang 1 and 2 to avoid confusion about the total number of languages. (The old Lang 1 and Lang 2 are still preserved, but are cleraly labeled "EXCLUDED" everywhere.)

***This change does not affect the crucial findings nor the general conclusion of this study***, but I believe it is a more accurate reflection of the nature of the project.

The older repository is preserved here: https://github.com/EunsunJou/Economy_RIPGLA_Old

# Economy_RIPGLA

This repository contains code and results files from my work on economy-based amendment of Robust Interpretive Parsing.
Specifically, it contains:

1. A python implementation of the RIP/OT-GLA, available as the module gla.py
2. An economy-based amendment to the algorithm (the RIP/ERC-GLA), also available in the gla.py module
3. A set of 66 randomly sampled abstract metrical stress languages, available in /languages
4. Files resulting from learning trials using the RIP/GLA and its amended version, available in /results

gla.py is intended to be loaded as a module in a separate python script.
worbench.py is an example script that loads the module and actually does the learning.
