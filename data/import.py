import os

import pandas as pd
from neo4j import GraphDatabase

env_dict = {}

with open('.env') as file:
    for line in file:
        # Strip leading and trailing whitespaces and split by '='
        key, value = line.strip().split('=')
        # Add the key-value pair to the dictionary
        env_dict[key] = value

neo4j_host = env_dict["NEO4J_URL"]
neo4j_user = env_dict["NEO4J_USER"]
neo4j_password = env_dict["NEO4J_PASS"]

driver = GraphDatabase.driver(neo4j_host, auth=(neo4j_user, neo4j_password))

create_full_text_index = """
CREATE FULLTEXT INDEX movie IF NOT EXISTS
FOR (n:Movie)
ON EACH [n.title]
"""

embedding_df = pd.read_csv("movie_embeddings.csv")
params = embedding_df.values
step = 1000

with driver.session() as session:
    data = session.run(create_full_text_index)

    for i in range(0, len(embedding_df), step):
        print(f"Importing {i} batch of embeddings")
        batch = [{'id': str(x[0]), 'title': x[1], 'embedding': x[2]}
                 for x in params[i:i+step]]
        session.run("""
        UNWIND $data AS row
        MERGE (m:Movie {movieId:row.id})
        SET m.title = row.title, m.embedding = apoc.convert.fromJsonList(row.embedding);
        """, {'data': batch})
print("Import completed")
