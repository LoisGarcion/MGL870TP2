from tempfile import template

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
import re

template_miner = TemplateMiner()

# define the log format and regex for parsing the logs
log_format = "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>"

# define a regex pattern based on the log format
log_pattern = re.compile(
    r"(?P<Label>\S+) "
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
    if i > 500000:
        break
    match = log_pattern.match(line)
    if match :
        log_content = match.group("Content")
        drain_parser.add_log_message(log_content)

templates = {}
parsed_events = []
for i,line in enumerate(logs):
    isalert = not line.strip().startswith("- ")
    isalert = int(isalert)
    if i % 10000 == 0:
        print(f"Parsing line {i}, isalert={isalert}")
    if i > 500000:
        break
    match = log_pattern.match(line)
    if match :
        log_content = match.group("Content")
        result = drain_parser.match(log_content)
        if result :
            template_id = result.cluster_id
            template_description = result.get_template()
            parsed_events.append({
                "Timestamp": match.group("Timestamp"),
                "Template ID": template_id,
                "Is Alert": isalert
            })
            if template_id not in templates:
                templates[template_id] = template_description
        else :
            print(f"No template found for line {i}: {line}")
    else :
        print(f"No match for line {i}: {line}")

parsed_df = pd.DataFrame(parsed_events)
parsed_df.to_csv("BGL_parsed.csv", index=False)
#dataframe de tous les events
templates_df = pd.DataFrame([
    {"Template ID": tid, "Template Description": desc}
    for tid, desc in templates.items()
])
templates_df.to_csv("BGL_templates.csv", index=False)

#GENERATION MATRICE POUR TOUTES LES 30 MINUTES
parsed_matrix_df = pd.read_csv("BGL_parsed.csv")
parsed_df['Timestamp'] = pd.to_datetime(parsed_df['Timestamp'].astype(int), unit='s')
parsed_df['Is Alert'] = parsed_df['Is Alert'].astype(int)
parsed_df.set_index('Timestamp', inplace=True)

alert_flag = parsed_df['Is Alert'].resample('30min').sum()

# Resample data into 30-minute intervals and count Template ID occurrences in each interval
template_counts = (
    parsed_df
    .groupby('Template ID')
    .resample('30min')
    .size()
    .unstack(fill_value=0)
)

template_counts = template_counts.transpose()
template_counts['Interval Start'] = template_counts.index
template_counts['Has Alert'] = alert_flag.values  # Now 'Has Alert' is 0 or 1
template_counts = template_counts[['Interval Start'] + [col for col in template_counts.columns if col not in ['Interval Start', 'Has Alert']] + ['Has Alert']]

template_counts.to_csv("BGL_template_counts_30min.csv", index=False)