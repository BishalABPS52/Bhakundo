"""Script to download historical Premier League data"""

import sys
import argparse
from pathlib import Path
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.etl_pipeline import ETLPipeline
from src.config import DATA_COLLECTION_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download historical Premier League data"
    )
    
    parser.add_argument(
        "--seasons",
        type=str,
        default=",".join(DATA_COLLECTION_CONFIG["seasons"]),
        help="Comma-separated list of seasons (e.g., '2019,2020,2021')"
    )
    
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing data with latest matches"
    )
    
    parser.add_argument(
        "--skip-xg",
        action="store_true",
        help="Skip xG data collection (faster but less comprehensive)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse seasons
    seasons = args.seasons.split(",")
    seasons = [s.strip() for s in seasons]
    
    logger.info("=" * 70)
    logger.info("Premier League Data Collection")
    logger.info("=" * 70)
    logger.info(f"Seasons to collect: {', '.join(seasons)}")
    logger.info(f"Update mode: {args.update}")
    logger.info(f"Skip xG data: {args.skip_xg}")
    logger.info("=" * 70)
    
    # Confirm before proceeding
    if not args.update:
        response = input("\nThis will download data from multiple APIs. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Operation cancelled")
            return
    
    # Initialize ETL pipeline
    logger.info("\nInitializing ETL pipeline...")
    pipeline = ETLPipeline()
    
    try:
        if args.update:
            # Update mode - only get latest season
            latest_season = seasons[-1]
            logger.info(f"\nUpdate mode: Collecting data for season {latest_season}")
            
            # Collect matches
            matches_df = pipeline.collect_historical_matches([latest_season])
            
            # Collect xG data if not skipped
            if not args.skip_xg:
                xg_df = pipeline.collect_xg_data([latest_season])
                merged_df = pipeline.merge_match_data(matches_df, xg_df)
            else:
                merged_df = matches_df
            
            # Load to database
            if not merged_df.empty:
                pipeline.load_to_database(merged_df)
            
            # Update FPL data
            pipeline.collect_fpl_data()
            
        else:
            # Full collection mode
            logger.info("\nStarting full data collection...")
            pipeline.run_full_pipeline(seasons=seasons)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Data collection completed successfully!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Verify data: Check database for collected matches")
        logger.info("2. Feature engineering: Run feature engineering scripts")
        logger.info("3. Model training: python scripts/train_models.py")
        
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error during data collection: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check API keys in .env file")
        logger.error("2. Verify internet connection")
        logger.error("3. Check database connection")
        sys.exit(1)


if __name__ == "__main__":
    main()
