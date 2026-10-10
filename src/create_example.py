import json
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/Base.csv", nrows=1)

X = df.drop(columns=["fraud_bool", "month"])
application = X.iloc[0].to_dict()

Path('examples/application.json').write_text(json.dumps(application, indent=2), encoding='utf-8')

print("Example saved!")