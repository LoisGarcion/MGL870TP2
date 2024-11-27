import re
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
config.load("../drain3.ini")
config.profiling_enabled = True
drain_parser = TemplateMiner(config=config)

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

log_pattern_no_node_repeat = re.compile(
    r"(?P<Label>\S+) "
    r"(?P<Timestamp>\d+) "
    r"(?P<Date>\d{4}\.\d{2}\.\d{2}) "
    r"(?P<Node>\S+) "
    r"(?P<Time>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+) "
    r"(?P<Type>\w,+) "
    r"(?P<Content>.*)"
)

templates = {}
parsed_events = []

with open("BGL.log", "r", encoding="utf-8") as f:
    logs = f.readlines()

with open("BGL.log", "r", encoding='utf-8') as f:
    logs = f.readlines()
for i,line in enumerate(logs):
    line = line.lower()
    if i % 10000 == 0:
        print(f"Matching line {i}")
    match = log_pattern.match(line)
    if match :
        log_content = match.group("Content")
        drain_parser.add_log_message(log_content)
    else :
        match = log_pattern_no_node_repeat.match(line)
        if match :
            log_content = match.group("Content")
            drain_parser.add_log_message(log_content)
        else :
            print(f"No match for line {i}: {line}")

templates = {}
parsed_events = []
for i,line in enumerate(logs):
    line = line.lower()
    isalert = not line.strip().startswith("- ")
    isalert = int(isalert)
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
                "Timestamp": match.group("Timestamp"),
                "Template ID": template_id,
                "Anomaly": isalert
            })
            if template_id not in templates:
                templates[template_id] = template_description
        else :
            print(f"No template found for line {i}: {line}")
    else :
        print(f"No match for line {i}: {line}")

print(f"Found {len(templates)} templates and {len(parsed_events)} parsed events")

parsed_df = pd.DataFrame(parsed_events)
print(len(parsed_df))
parsed_df.to_csv("BGL_parsed.csv", index=False)

parsed_df["Timestamp"] = pd.to_datetime(parsed_df["Timestamp"].astype(int), unit="s")
parsed_df["Anomaly"] = parsed_df["Anomaly"].astype(int)
parsed_df.set_index("Timestamp", inplace=True)

# ERROR DETECTION
def count_within_30min_or_first_alert(group):
    if group.empty:
        return 0
    group = group[group.index < (group.index[0] + pd.Timedelta(minutes=30))]
    return group['Anomaly'].cumsum().eq(0).sum()

template_counts_30min = parsed_df.groupby("Template ID").resample("30min").apply(
    count_within_30min_or_first_alert
).unstack(fill_value=0).transpose()
template_counts_30min["Anomaly"] = parsed_df["Anomaly"].resample("30min").sum().astype(bool).astype(int)
template_counts_30min["Datetime"] = template_counts_30min.index
template_counts_30min = template_counts_30min[["Datetime"] + [col for col in template_counts_30min.columns if col != "Datetime"]]

non_zero_columns = [col for col in template_counts_30min.columns if col not in ["Datetime", "Anomaly"]]
template_counts_30min = template_counts_30min[(template_counts_30min[non_zero_columns].sum(axis=1) > 0)]

template_counts_30min.to_csv("BGL_template_counts_error_detection.csv", index=False)

# ERROR PREDICTION
def count_within_10min_or_first_alert(group):
    if group.empty:
        return 0
    group = group[group.index < (group.index[0] + pd.Timedelta(minutes=10))]
    return group['Anomaly'].cumsum().eq(0).sum()

template_counts_10min = parsed_df.groupby("Template ID").resample("30min").apply(
    count_within_10min_or_first_alert
).unstack(fill_value=0).transpose()
template_counts_10min["Anomaly"] = parsed_df["Anomaly"].resample("30min").sum().astype(bool).astype(int)
template_counts_10min["Datetime"] = template_counts_10min.index
template_counts_10min = template_counts_10min[["Datetime"] + [col for col in template_counts_10min.columns if col != "Datetime"]]

non_zero_columns = [col for col in template_counts_10min.columns if col not in ["Datetime", "Anomaly"]]
template_counts_10min = template_counts_10min[(template_counts_10min[non_zero_columns].sum(axis=1) > 0)]

template_counts_10min.to_csv("BGL_template_counts_error_prediction.csv", index=False)