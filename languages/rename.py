import os
import re

files = os.listdir()


for f in files:
    fname = re.sub(r"\.txt_data", "_data", f)
    os.rename(f, fname)


