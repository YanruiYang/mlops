import os
import pandas as pd

#path to raw data
path = "../../data/bronze/"

#the text files in the raw data are (1) named after the hacker group and (2) all end in list.txt
cut_length = len("list.txt") + 1

#list all the files in the path
files = os.listdir(path)

#create a dataframe that will hold all my data
dga_df = pd.DataFrame()


#loop through each file
for file in files:
    
    #if it ends with .txt, it was pulled from https://data.mendeley.com/datasets/y8ph45msv8/1
    if file.endswith(".txt"):

        #the files contain a domain per line. Go through each line and append to domains list
        with open(f"{path}/{file}") as f:
            domains = []
            line = f.readline()
            while line != "":
                line = f.readline()
                if line != "":
                    domains.append(line.strip("\n"))
        
        #convert domains list into dataframe and add features isdga and actor                    
        df = pd.DataFrame(domains, columns=["domains"])
        df['isdga'] = 1
        df['actor'] = file[:-cut_length]

        #concat to final df
        dga_df = pd.concat([dga_df, df], axis=0)
        
    #else it was pulled from https://majestic.com/reports/majestic-million
    elif file.endswith(".csv"):
        #read csv into pandas DF, add features isdga and actor, drop others, concat to dga_df
        df = pd.read_csv(f"{path}/{file}", index_col=0, usecols=["Domain"])
        df['isdga'] = 0
        df['actor'] = "None"
        dga_df = pd.concat([dga_df, df], axis=0)

#shuffle dataframe
dga_df = dga_df.sample(frac=1, random_state=42)
dga_df.reset_index(inplace=True)
dga_df.drop('index', inplace=True, axis=1)


#feature engineering

def entropy(domain, base=2):
    entropy = 0.0
    length = len(domain)

    occ = {}
    for c in domain:
        if not c in occ:
            occ[c] = 0
        occ[c] += 1

    for (k, v) in occ.items():
        p = float(v) / float(length)
        entropy -= p * math.log(p, base)

    return entropy


def length(domain):
    return len(domain)


dga_df['entropy'] = [entropy(domain) for domain in dga_df['domains']]
dga_df['length'] = [length(domain) for domain in dga_df['domains']]




#READ THE PAPER TO GET THE REST OF THE SUGGESTED FEATURES




#save the reformatted data
dest_dir = "../../data/silver"
os.makedirs(dest_dir, exist_ok=True)
dga_df.to_csv(f"{dest_dir}/DGAs.csv")
