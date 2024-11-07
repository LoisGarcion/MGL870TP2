from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
import re

template_miner = TemplateMiner()

# define the log format and regex for parsing the logs
log_format = "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>"

# define a regex pattern based on the log format
log_pattern = re.compile(
    r"(?P<Label>[-\d]+) "
    r"(?P<Timestamp>\d+) "
    r"(?P<Date>\d{4}\.\d{2}\.\d{2}) "
    r"(?P<Node>[\w:-]+) "
    r"(?P<Time>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+) "
    r"(?P<NodeRepeat>[\w:-]+) "
    r"(?P<Type>\w+) "
    r"(?P<Component>\w+) "
    r"(?P<Level>\w+) "
    r"(?P<Content>.*)"
)
config = TemplateMinerConfig()
config.load("drain3.ini")
config.profiling_enabled = True
drain_parser = TemplateMiner(config=config)


with open("BGL/BGL.log", "r", encoding='utf-8') as f:
    logs = f.readlines()
for i,line in enumerate(logs):
    if i % 10000 == 0:
        print(f"Matching line {i}")
    match = log_pattern.match(line)
    if match :
        log_content = match.group("Content")
        drain_parser.add_log_message(log_content)

parsed_events = []
for i,line in enumerate(logs):
    isalert = not line.startswith("-")
    if i % 10000 == 0:
        print(f"Parsing line {i}")
    match = log_pattern.match(line)
    if match :
        log_content = match.group("Content")
        result = drain_parser.match(log_content)
        if result :
            template_id = result.cluster_id
            template_description = result.get_template()
            parsed_events.append({
                "Template ID": template_id,
                "Template Description": template_description,
                "Log Content": log_content,
                "Is Alert": isalert
            })

parsed_df = pd.DataFrame(parsed_events)
parsed_df.to_csv("BGL_parsed.csv", index=False)

#IL FAUDRA DECOUPER MON TABLEAU PAR rapport à la date ou pour une session et compter le nombre pour chaque evenement et si on a une alerte ou non
#C'est cette matrice qui par la suite va nous servir à entrainer le modele avec notre sur discord