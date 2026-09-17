import os
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine

def load_data():
   # get password to database
    load_dotenv()
    db_password = os.getenv("DB_PASSWORD")

    engine = create_engine(
        f"postgresql://ml_students:{db_password}@dpg-d35pib0dl3ps7394mc4g-a.oregon-postgres.render.com:5432/beam_neb0"
    )
    # Pull raw data from database for latest training run
    data = pd.read_sql("SELECT * FROM ml.trial_snapshot_latest", engine)
    # Save a copy of raw data for model training
    data.to_csv("data/01_raw/trials.csv", index=False)

    return data