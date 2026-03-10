"""
Download official Premier League team logos for 2025-26 season
"""

import requests
from pathlib import Path
import time

# Team logos mapping - using official sources
TEAM_LOGOS = {
    "Arsenal FC": "https://resources.premierleague.com/premierleague/badges/50/t3.png",
    "Aston Villa FC": "https://resources.premierleague.com/premierleague/badges/50/t7.png",
    "AFC Bournemouth": "https://resources.premierleague.com/premierleague/badges/50/t91.png",
    "Brentford FC": "https://resources.premierleague.com/premierleague/badges/50/t94.png",
    "Brighton & Hove Albion FC": "https://resources.premierleague.com/premierleague/badges/50/t36.png",
    "Burnley FC": "https://resources.premierleague.com/premierleague/badges/50/t90.png",
    "Chelsea FC": "https://resources.premierleague.com/premierleague/badges/50/t8.png",
    "Crystal Palace FC": "https://resources.premierleague.com/premierleague/badges/50/t31.png",
    "Everton FC": "https://resources.premierleague.com/premierleague/badges/50/t11.png",
    "Fulham FC": "https://resources.premierleague.com/premierleague/badges/50/t54.png",
    "Leeds United FC": "https://resources.premierleague.com/premierleague/badges/50/t2.png",
    "Liverpool FC": "https://resources.premierleague.com/premierleague/badges/50/t14.png",
    "Manchester City FC": "https://resources.premierleague.com/premierleague/badges/50/t43.png",
    "Manchester United FC": "https://resources.premierleague.com/premierleague/badges/50/t1.png",
    "Newcastle United FC": "https://resources.premierleague.com/premierleague/badges/50/t4.png",
    "Nottingham Forest FC": "https://resources.premierleague.com/premierleague/badges/50/t17.png",
    "Sunderland AFC": "https://resources.premierleague.com/premierleague/badges/50/t29.png",
    "Tottenham Hotspur FC": "https://resources.premierleague.com/premierleague/badges/50/t6.png",
    "West Ham United FC": "https://resources.premierleague.com/premierleague/badges/50/t21.png",
    "Wolverhampton Wanderers FC": "https://resources.premierleague.com/premierleague/badges/50/t39.png",
}

def download_logos():
    """Download team logos to frontend public folder"""
    logos_dir = Path(__file__).parent.parent / 'frontend' / 'public' / 'team-logos'
    logos_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading Premier League 2025-26 team logos...")
    
    for team, url in TEAM_LOGOS.items():
        # Create filename from team name
        filename = team.replace(' ', '_').replace('&', 'and') + '.png'
        filepath = logos_dir / filename
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✅ {team}")
            else:
                print(f"⚠️  {team} - Status: {response.status_code}")
            
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"❌ {team} - Error: {e}")
    
    print(f"\n✨ Logos saved to: {logos_dir}")

if __name__ == '__main__':
    download_logos()
