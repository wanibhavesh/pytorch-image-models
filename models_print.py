import timm
import pandas as pd

# Get all available model names
all_models = timm.list_models()

# Create a DataFrame
df = pd.DataFrame(all_models, columns=["Model Name"])

# Save to CSV
df.to_csv("timm_model_list.csv", index=False)

print("CSV saved as timm_model_list.csv")
