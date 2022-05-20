import os
import shutil


# List of folders (i.e., "hypo03") 
folders = []

for folder in os.listdir():
    if folder[:4] == "hypo":
        folders.append(folder)


# Absolute path of current script
pathname = os.path.dirname(os.path.abspath(__file__))


# List of absolute paths (i.e., "D:\\Dropbox\\sample\\hypo03")
paths = []
# List of tuples of (<folder name>, <folder path>)
folders_paths = []


for f in folders:
    folder_path = pathname + "\\" + f
    paths.append(folder_path)
    folders_paths.append((f, folder_path))


### CHUNK 3: COUNT THE SUCCESSES AND FAILURES FOR EACH LANG ###

# Count the num of success and failures and write them into csv file
outfile = open('nonnoisy_sortinfo.csv', 'w')

for f in folders_paths:
    success_folder = f[1]+"\\success\\"
    fail_folder = f[1]+"\\fail\\"
    outfile.write(",".join([f[0], "success", str(len(os.listdir(success_folder)))]))
    outfile.write("\n")
    outfile.write(",".join([f[0], "fail", str(len(os.listdir(fail_folder)))]))
    outfile.write("\n")

outfile.close()



# The following is a danger zone: it makes directories and copy files,
# So I protect it from accidentally running by using exit()

exit()


### CHUNK 1: MAKE 'SUCCESS' AND 'FAIL' SUBDIRECTORIES ###

'''
# Making a 'success' subfolder and a 'fail' subfolder
# (You need to make a folder before copying into it)
for p in paths:
    os.mkdir(p+"\\success\\")
    os.mkdir(p+"\\fail\\")
'''



### CHUNK 2: COPY SUCCESS AND FAIL FILES TO RESPECTIVE SUBDIRS ###

# Dictionaries for counting
success = {}
fail = {}

# Make an empty list for each folder. Append files to that folder.
for p in paths:
    success[p] = []
    fail[p] = []

# 
for p in paths:
    # Collect all txt result files into a list
    result_files = []
    if len(os.listdir(p)) != 0:
        for q in os.listdir(p):
            abs_path = p+"\\"+q
            result_files.append((q, abs_path))
    # Sort out successful and filed learnings
    # Uses string search to detect expressions unique to successful/failed learnings
    for f in result_files:
        if f[0][-4:] == ".txt":
            file = open(f[1], 'r')
            ftext = file.read()
            noerror = ftext.find("No errors found in evaluation")
            yeserror = ftext.find("words not (fully) learned in evaluation")
            if noerror != -1 and yeserror == -1: # index is -1 if no match found
                success[p].append(f)
            elif noerror == -1 and yeserror != -1:
                fail[p].append(f)
            # Raise error if search is not exclusively yes or no
            # (This hasn't happened yet)
            else:
                raise ValueError("Failed to sort "+f+" -- please check file")

# Move the success files to success subfolder
for p in success.keys():
    for f in success[p]:
        success_subdir = p+"\\success\\"+f[0]
        shutil.copy(f[1], success_subdir)

# Move the fail files to fail subfolder
for p in fail.keys():
    for f in fail[p]:
        fail_subdir = p+"\\fail\\"+f[0]
        shutil.copy(f[1], fail_subdir)




