from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os


class GBQContextManager:
    def __init__(self):
        self.client = bigquery.Client()
        self.table_id = os.getenv("RUN_TABLE_ID")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def create_runs_table(self) -> None:
        schema = [
            bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("date", "STRING", mode="REQUIRED"),
			bigquery.SchemaField("user", "STRING", mode="REQUIRED"),
			bigquery.SchemaField("embedding_features", "STRING", mode="REQUIRED"),
			bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
        ]
        table = bigquery.Table(self.table_id, schema=schema)
        self.client.create_table(table)

    def check_table_exists(self) -> None:
        try:
            self.client.get_table(self.table_id)
            print("Table {} already exists.".format(self.table_id))
        except NotFound:
            print("Table {} is not found. Recreating it.".format(self.table_id))
            self.create_runs_table()

    def append_run(self, tracker):
        self.check_table_exists()

        rows_to_insert = [
            {"run_id": tracker.run_id,
             "date": tracker.date,
             "user": tracker.user,
             "embedding_features": str(tracker.embedding_features),
			 "model": tracker.model
             }
            ]

        errors = self.client.insert_rows_json(self.table_id, rows_to_insert)

        if errors:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
