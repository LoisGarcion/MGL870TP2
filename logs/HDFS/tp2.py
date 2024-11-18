from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
import re
from collections import defaultdict


######
anomalies = []
def filtrer_lignes_avec_anomaly():

    # Ouvrir le fichier d'entrée en mode lecture
    with open("HDFS_v1/preprocessed/anomaly_label.csv", 'r') as fichier:
            # Lire chaque ligne du fichier
            for ligne in fichier:
                # Diviser la ligne en deux parties : blk et état
                blk, statut = ligne.strip().split(",")
                # Vérifier si le statut est 'Anomaly'
                if statut.strip() == "Anomaly":
                    # Ajouter le blk à la liste
                    anomalies.append(blk)
                    
filtrer_lignes_avec_anomaly()

with open("allAnomaly.csv", "r") as file4:
    logs9 = file4.readlines()
def matchBlks(myBlk):
    for line in anomalies:
                    if(line == myBlk):
                        return "ok"
    return None
########

template_miner = TemplateMiner()

log_format = "<Date> <Time> <Pid> <Level> <Component> <Content>"


log_pattern = re.compile(
    r'^(?P<Date>\d{6})\s+'
    r'(?P<Time>\d{6})\s+'
    r'(?P<Pid>\d+)\s+'
    r'(?P<Level>[A-Z]+)\s+'
    r'(?P<Component>[\w\.\$]+):\s+'
    r'(?P<Content>.*)'
)

blk_pattern  = re.compile(r'blk_[-\d]+')
templ_pattern = re.compile(r'E.*')

config = TemplateMinerConfig()
config.load("drain3.ini")
config.profiling_enabled = True
drain_parser = TemplateMiner(config=config)
ligne = 0
myTemplates = []
existe = False
print("starting")   


with open("fileBlk.txt", 'w') as file2:

    with open("myTemplates.txt", 'w') as file:

        with open("HDFS_v1/HDFS.log", "r", encoding='utf-8') as f:
            logs = f.readlines()
        for i,line in enumerate(logs):
            match = log_pattern.match(line)
            if match :
                blk = match.group("Content")
                match2 = blk_pattern.findall(blk)
                template = re.sub(r'(\d+\.\d+\.\d+\.\d+:\d+)', '<*>', blk)
                template = re.sub(r'[-_]?\d+', '<*>', template)
                template = re.sub(r'<\*>[^a-zA-Z0-9]<\*>', '<*>', template)
                template = re.sub(r'(<\*>)+', '<*>', template)
                template = re.sub(r'<\*>[^a-zA-Z0-9]<\*>', '<*>', template)
                template = template.replace(', <*>', '<*>')
                                
                                
                template = re.sub(r'(<\*>)+', '<*>', template)

                template = re.sub(r'blk_<\*>', 'blk<*>', template)
                template = re.sub(r'/\S+', '<*>', template)
                template = re.sub(r'received exception.*', 'received exception <*>', template)
                template = re.sub(r'Exception in receiveBlock for block.*',   'Exception in receiveBlock for block <*>', template)
                template = re.sub(r'java.*',   '<*>', template)
                j=0
                for k in myTemplates :
                    if(k == template):
                        existe = True
                        file2.write(match2[0] + '\t' + f"E{j}" + '\n')
                        j=0
                        break
                    j+=1
                if(existe == False):
                    file.write(f"E{ligne}" + '\t' + template + '\n')
                    file2.write(match2[0] + '\t' + f"E{ligne}" + '\n')
                    myTemplates.append(template)
                    ligne += 1  
                    
                existe = False   
print("finished")  


with open("fileBlk.txt", "r") as file:
    log_entries = [line.strip().split() for line in file if line.strip()]
blk_dict = defaultdict(list)
for blk, exx in log_entries:
    blk_dict[blk].append(exx)
with open("fileBlk2.csv", 'w') as file3:
    with open("resultat2.csv", "w") as file2:
        file2.write("BlockId,Label,E0,E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15,E16,E17,E18,E19,E20,E21,E22,E23,E24,E25,E26,E27,E28,E29"+ '\n')
        for blk, exx_list in blk_dict.items():
            if(matchBlks(blk) == None):
                ligne = blk + ",Normal,"
            else:
                ligne = blk + ",Anomaly,"
            exx_str = ",".join(exx_list)
            file3.write(f'{blk},"[{exx_str}]"' + '\n')
            myStr = str(f'{blk},"[{exx_str}]"' + '\n')
            for k in range(30):
                nombre_occurrences = myStr.count(f'E{k},') + myStr.count(f'E{k}]')
                ligne += str(nombre_occurrences) + ","
            file2.write(ligne[:-1] + '\n')
            ligne = ""





                
                

