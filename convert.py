import os
mypath = '../datasets/supermarket_datasetV2/labels'
labelfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
print(labelfiles)


for file in labelfiles:
    with open(os.path.join(mypath, file), 'r') as f:
        lines = f.readlines()
    
    splitted_lines = [['0'] + l.split(' ')[1::] for l in lines]
    new_lines = [' '.join(splitted_line) for splitted_line in splitted_lines]
    print(new_lines)

    with open('labels/' + file, 'w') as f:
        f.writelines(new_lines)