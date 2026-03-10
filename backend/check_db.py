"""Check database connection and tables"""
from database import engine, Base, Prediction, Actual
from sqlalchemy import text, inspect

print("Checking database connection...")

with engine.connect() as conn:
    # Check which database we're connected to
    result = conn.execute(text('SELECT current_database()'))
    db_name = result.fetchone()[0]
    print(f"Connected to database: {db_name}")
    
    # Check existing tables
    result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
    tables = [r[0] for r in result.fetchall()]
    print(f"\nExisting tables ({len(tables)}):")
    for table in tables:
        print(f"  - {table}")

# Create tables if they don't exist
print("\nCreating tables...")
Base.metadata.create_all(engine)
print("✓ Tables created")

# Verify
inspector = inspect(engine)
all_tables = inspector.get_table_names()
print(f"\nTables after creation ({len(all_tables)}):")
for table in all_tables:
    print(f"  - {table}")
    
if 'prediction' in all_tables and 'actual' in all_tables:
    print("\n✅ SUCCESS: prediction and actual tables are ready!")
else:
    print("\n❌ ERROR: Tables not created properly")
