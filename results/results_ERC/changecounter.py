import os
import shutil
import re

# List of folders (i.e., "hypo03") 
folders = []

for folder in os.listdir():
    if folder[:4] == "hypo":
        folders.append(folder)


# Absolute path of current script
pathname = os.path.dirname(os.path.abspath(__file__))

# List of absolute paths (i.e., "D:\\Dropbox\\sample\\hypo03")
success_paths = []
# List of tuples of (<folder name>, <folder path>)
folders_paths = []

for f in folders:
    folder_path = pathname + "\\" + f + "\\success\\"
    success_paths.append((f, folder_path))
    folders_paths.append((f, folder_path))

change_pattern = re.compile(r"Grammar changed (\d+)/62000 times")

# Count the num of success and failures and write them into csv file
outfile = open('nonnoisy_changeinfo.csv', 'w')

success_changecounts = {}

for p in success_paths:
    files = os.listdir(p[1])
    
    changenums = []

    for f in files:
        filedir = p[1]+"\\"+f
        fh = open(filedir, 'r')
        txt = fh.read()
        changenum = re.search(change_pattern, txt)[1]
        changenums.append(changenum)
    
    if len(changenums) > 0:
        changestr = ",".join(changenums)
        success_changecounts[p[0]] = changestr
    else:
        success_changecounts[p[0]] = "N/A (No successful trial)"
    
for path, counts in success_changecounts.items():
    out = path+","+counts+"\n"
    outfile.write(out)
    
outfile.close()
