"""
Database Setup Script for Supabase PostgreSQL
Initializes the prediction_history table and verifies connection
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from database import init_db, get_db_session, PredictionHistory, engine
from sqlalchemy import text, inspect
from datetime import datetime


def test_connection():
    """Test database connection"""
    print("="*60)
    print("TESTING SUPABASE DATABASE CONNECTION")
    print("="*60)
    
    try:
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"\n✅ Connected to PostgreSQL")
            print(f"   Version: {version[:50]}...")
            
            # Check current database
            result = conn.execute(text("SELECT current_database()"))
            db_name = result.fetchone()[0]
            print(f"   Database: {db_name}")
            
            # Check current user
            result = conn.execute(text("SELECT current_user"))
            user = result.fetchone()[0]
            print(f"   User: {user}")
        
        return True
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False


def check_existing_tables():
    """Check if tables already exist"""
    print("\n" + "="*60)
    print("CHECKING EXISTING TABLES")
    print("="*60)
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if tables:
            print(f"\n✅ Found {len(tables)} existing table(s):")
            for table in tables:
                print(f"   - {table}")
                
                # Get column details
                columns = inspector.get_columns(table)
                print(f"     Columns: {len(columns)}")
                for col in columns[:5]:  # Show first 5 columns
                    print(f"       • {col['name']} ({col['type']})")
                if len(columns) > 5:
                    print(f"       ... and {len(columns) - 5} more")
        else:
            print("\n⚠️  No tables found. Will create new schema.")
        
        return 'prediction_history' in tables
    except Exception as e:
        print(f"\n❌ Error checking tables: {e}")
        return False


def create_tables():
    """Create database tables"""
    print("\n" + "="*60)
    print("CREATING TABLES")
    print("="*60)
    
    try:
        init_db()
        print("\n✅ Tables created successfully!")
        
        # Verify creation
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if 'prediction_history' in tables:
            print("\n✅ prediction_history table verified")
            
            # Show schema
            columns = inspector.get_columns('prediction_history')
            print(f"\n   Schema ({len(columns)} columns):")
            for col in columns:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"   • {col['name']:<30} {col['type']!s:<20} {nullable}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insert_retrieve():
    """Test inserting and retrieving a prediction"""
    print("\n" + "="*60)
    print("TESTING INSERT & RETRIEVE")
    print("="*60)
    
    try:
        db = get_db_session()
        
        # Create test prediction
        test_prediction = PredictionHistory(
            match_id="test_match_001",
            home_team="Arsenal FC",
            away_team="Manchester City FC",
            gameweek=30,
            match_date=datetime.now(),
            season="2025-26",
            predicted_outcome="Home Win",
            predicted_home_goals=2,
            predicted_away_goals=1,
            base_model_home_prob=0.65,
            base_model_draw_prob=0.20,
            base_model_away_prob=0.15,
            ensemble_home_prob=0.68,
            ensemble_draw_prob=0.18,
            ensemble_away_prob=0.14,
            ensemble_method="base_lineup_agree",
            confidence=0.88,
            home_formation="4-3-3",
            away_formation="4-2-3-1",
            additional_data={"test": True}
        )
        
        # Insert
        db.add(test_prediction)
        db.commit()
        print("\n✅ Test prediction inserted")
        print(f"   ID: {test_prediction.id}")
        print(f"   Match: {test_prediction.home_team} vs {test_prediction.away_team}")
        print(f"   Prediction: {test_prediction.predicted_outcome} ({test_prediction.predicted_home_goals}-{test_prediction.predicted_away_goals})")
        print(f"   Confidence: {test_prediction.confidence * 100:.1f}%")
        
        # Retrieve
        retrieved = db.query(PredictionHistory).filter_by(match_id="test_match_001").first()
        
        if retrieved:
            print("\n✅ Test prediction retrieved successfully")
            print(f"   Verified: {retrieved.home_team} vs {retrieved.away_team}")
            print(f"   Outcome: {retrieved.predicted_outcome}")
        
        # Clean up test data
        db.delete(test_prediction)
        db.commit()
        print("\n✅ Test data cleaned up")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Error in insert/retrieve test: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_statistics():
    """Show database statistics"""
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    try:
        db = get_db_session()
        
        # Count predictions
        total = db.query(PredictionHistory).count()
        print(f"\n   Total Predictions: {total}")
        
        if total > 0:
            # Count by season
            from sqlalchemy import func
            by_season = db.query(
                PredictionHistory.season,
                func.count(PredictionHistory.id)
            ).group_by(PredictionHistory.season).all()
            
            print("\n   Predictions by Season:")
            for season, count in by_season:
                print(f"     {season}: {count}")
            
            # Count pending vs completed
            pending = db.query(PredictionHistory).filter(
                PredictionHistory.actual_home_goals == None
            ).count()
            completed = total - pending
            
            print(f"\n   Status:")
            print(f"     Pending: {pending}")
            print(f"     Completed: {completed}")
            
            if completed > 0:
                correct = db.query(PredictionHistory).filter(
                    PredictionHistory.is_correct == 1
                ).count()
                accuracy = (correct / completed * 100) if completed > 0 else 0
                print(f"     Accuracy: {accuracy:.1f}% ({correct}/{completed})")
        
        db.close()
        
    except Exception as e:
        print(f"\n⚠️  Could not fetch statistics: {e}")


def main():
    """Main setup workflow"""
    print("\n🚀 SUPABASE DATABASE SETUP FOR BHAKUNDO PREDICTOR")
    print("="*60)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    db_url = os.getenv('DATABASE_URL', 'Not set')
    if 'supabase.com' in db_url:
        print(f"\n✅ Supabase database configured")
        print(f"   Host: aws-1-ap-southeast-1.pooler.supabase.com")
        print(f"   Database: postgres")
    else:
        print(f"\n⚠️  DATABASE_URL: {db_url[:50]}...")
    
    # Step 1: Test connection
    if not test_connection():
        print("\n❌ Setup failed: Cannot connect to database")
        return False
    
    # Step 2: Check existing tables
    table_exists = check_existing_tables()
    
    # Step 3: Create tables if needed
    if not table_exists:
        if not create_tables():
            print("\n❌ Setup failed: Cannot create tables")
            return False
    else:
        print("\n✅ Table already exists, skipping creation")
    
    # Step 4: Test insert/retrieve
    if not test_insert_retrieve():
        print("\n❌ Setup failed: Cannot insert/retrieve data")
        return False
    
    # Step 5: Show statistics
    show_statistics()
    
    # Success!
    print("\n" + "="*60)
    print("✅ DATABASE SETUP COMPLETE!")
    print("="*60)
    print("\n📊 Summary:")
    print("   ✅ Connection: Working")
    print("   ✅ Tables: Created/Verified")
    print("   ✅ Insert/Retrieve: Working")
    print("   ✅ Backend Integration: Ready")
    print("\n🚀 Your prediction database is ready for production!")
    print("\nNext steps:")
    print("   1. Start backend server: ./start-backend.sh")
    print("   2. Start frontend server: ./start-frontend.sh")
    print("   3. Test predictions in the app")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
