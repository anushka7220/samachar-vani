from app.database import create_db_and_tables
from app import models   # VERY IMPORTANT (loads the table model)

create_db_and_tables()

print("Database and tables created successfully")