from drain3 import TemplateMiner
import pandas as pd

template_miner = TemplateMiner()

detailed_counts = {}
alert_counts = {}
non_alert_counts = {}

with open("BGL/BGL.log", "r", encoding='utf-8') as log_file:
    for i,line in enumerate(log_file):
        if i % 10000 == 0:
            print(f"Processing line {i}")
        is_alert = not line.startswith("-")

        result = template_miner.add_log_message(line.strip())

        template_id = result["cluster_id"]
        template_description = result["template_mined"]

        if template_id not in detailed_counts:
            detailed_counts[template_id] = {
                "Description": template_description,
                "Alert Count": 0,
                "Non-Alert Count": 0
            }

        if is_alert:
            detailed_counts[template_id]["Alert Count"] += 1
            alert_counts[template_id] = alert_counts.get(template_id, 0) + 1
        else:
            detailed_counts[template_id]["Non-Alert Count"] += 1
            non_alert_counts[template_id] = non_alert_counts.get(template_id, 0) + 1

detailed_matrix_df = pd.DataFrame.from_dict(detailed_counts, orient="index")
detailed_matrix_df.index.name = "Template ID"

transposed_matrix_df = pd.DataFrame({
    "Alert": pd.Series(alert_counts),
    "Non-Alert": pd.Series(non_alert_counts)
}).fillna(0).astype(int).T

detailed_matrix_df.to_csv("detailed_log_matrix.csv")
transposed_matrix_df.to_csv("alert_nonalert_matrix.csv")

print("Detailed Matrix:")
print(detailed_matrix_df)
print("\nTransposed Matrix:")
print(transposed_matrix_df)
