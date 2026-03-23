#!/usr/bin/env python3
"""
Fetch and display Premier League 2025-26 season standings
Uses football-data.org API with authentication
"""
import requests
import json
import os
from datetime import datetime
from typing import List, Dict

def get_pl_standings() -> List[Dict]:
    """Fetch PL standings for 2025-26 season"""
    api_key = os.getenv('FOOTBALL_API_KEY', '')
    
    if not api_key:
        print("❌ Error: FOOTBALL_API_KEY environment variable not set")
        print("   Add it to backend/.env or set: export FOOTBALL_API_KEY='your-key'")
        return []
    
    url = "https://api.football-data.org/v4/competitions/2021/standings"
    headers = {'X-Auth-Token': api_key}
    params = {'season': 2025}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        
        if resp.status_code != 200:
            print(f"❌ API Error: {resp.status_code}")
            print(f"   {resp.json().get('message', 'Unknown error')}")
            return []
        
        data = resp.json()
        standings_data = data.get('standings', [])
        
        if not standings_data:
            print("❌ No standings data found")
            return []
        
        # Get TOTAL standings (index 0)
        total_standings = standings_data[0].get('table', [])
        
        # Convert to standardized format
        standings = []
        for entry in total_standings:
            team_info = entry.get('team', {})
            standing = {
                'position': entry.get('position'),
                'team': team_info.get('name'),
                'played': entry.get('playedGames'),
                'won': entry.get('won'),
                'drawn': entry.get('draw'),
                'lost': entry.get('lost'),
                'goals_for': entry.get('goalsFor'),
                'goals_against': entry.get('goalsAgainst'),
                'goal_difference': entry.get('goalDifference'),
                'points': entry.get('points'),
                'form': entry.get('form', ''),
                'crest': team_info.get('crest')
            }
            standings.append(standing)
        
        return standings
    
    except Exception as e:
        print(f"❌ Error fetching standings: {e}")
        return []

def print_standings_table(standings: List[Dict]):
    """Print standings in a nice table format"""
    if not standings:
        print("No standings data available")
        return
    
    print("\n" + "="*120)
    print("PREMIER LEAGUE 2025-26 SEASON STANDINGS".center(120))
    print("="*120)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*120)
    
    # Print header
    print(f"{'Pos':<4} {'Team':<30} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<4} {'Pts':<4} {'Form':<20}")
    print("-"*120)
    
    # Print each team
    for team in standings:
        pos = team['position']
        name = team['team']
        p = team['played']
        w = team['won']
        d = team['drawn']
        l = team['lost']
        gf = team['goals_for']
        ga = team['goals_against']
        gd = team['goal_difference']
        pts = team['points']
        form = team['form'] if team['form'] else 'N/A'
        
        # Format row with zone highlighting
        zone = ""
        if pos <= 4:
            zone = "🔵 CL"  # Champions League
        elif pos == 5:
            zone = "🟠 EL"  # Europa League
        elif pos >= 18:
            zone = "🔴 REL"  # Relegation
        
        row = f"{pos:<4} {name:<30} {p:<3} {w:<3} {d:<3} {l:<3} {gf:<3} {ga:<3} {gd:<4} {pts:<4} {form:<20}"
        if zone:
            row += f" {zone}"
        
        print(row)
    
    print("="*120)
    print("\nLegend:")
    print("  P=Played, W=Won, D=Drawn, L=Lost, GF=Goals For, GA=Goals Against, GD=Goal Difference, Pts=Points")
    print("  🔵 CL = Champions League (1-4)")
    print("  🟠 EL = Europa League (5)")
    print("  🔴 REL = Relegation (18-20)")
    print("="*120)

def export_json(standings: List[Dict], filename: str = "pl_standings_2025_26.json"):
    """Export standings to JSON file"""
    if not standings:
        print("No data to export")
        return
    
    output = {
        'season': '2025-26',
        'competition': 'Premier League',
        'timestamp': datetime.now().isoformat(),
        'total_teams': len(standings),
        'standings': standings
    }
    
    filepath = f"/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/{filename}"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✅ Standings exported to: {filename}")
    except Exception as e:
        print(f"❌ Error exporting: {e}")

def print_team_stats_summary(standings: List[Dict]):
    """Print summary statistics"""
    if not standings:
        return
    
    print("\n" + "="*60)
    print("LEAGUE STATISTICS".center(60))
    print("="*60)
    
    total_goals = sum(t['goals_for'] for t in standings)
    total_matches = sum(t['played'] for t in standings) // 2  # Each match counted twice
    avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 0
    
    print(f"Total Teams:          {len(standings)}")
    print(f"Total Matches Played: {total_matches}")
    print(f"Total Goals Scored:   {total_goals}")
    print(f"Average Goals/Match:  {avg_goals_per_match:.2f}")
    print(f"Total Points:         {sum(t['points'] for t in standings)}")
    print(f"Biggest Win:          {max(t['goal_difference'] for t in standings)}")
    print(f"Biggest Loss:         {min(t['goal_difference'] for t in standings)}")
    
    # Top scorers and best defense
    sorted_by_gf = sorted(standings, key=lambda x: x['goals_for'], reverse=True)
    sorted_by_ga = sorted(standings, key=lambda x: x['goals_against'])
    
    print(f"\nTop Scorer:           {sorted_by_gf[0]['team']} ({sorted_by_gf[0]['goals_for']} goals)")
    print(f"Best Defense:         {sorted_by_ga[0]['team']} ({sorted_by_ga[0]['goals_against']} goals against)")
    
    print("="*60)

if __name__ == '__main__':
    import sys
    
    # Set API key from environment or .env file
    if not os.getenv('FOOTBALL_API_KEY'):
        try:
            from dotenv import load_dotenv
            # Load from backend/.env
            env_path = "/home/bishal-shrestha/MyProjects/PL model/bhakundo-predictor/backend/.env"
            load_dotenv(env_path)
        except:
            pass
    
    print("\n🔄 Fetching Premier League 2025-26 standings...")
    standings = get_pl_standings()
    
    if standings:
        print_standings_table(standings)
        print_team_stats_summary(standings)
        
        # Export to JSON
        export_json(standings)
        
        # Option to export to CSV
        if len(sys.argv) > 1 and sys.argv[1] == '--csv':
            import csv
            try:
                csv_file = "pl_standings_2025_26.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['position', 'team', 'played', 'won', 'drawn', 'lost', 
                                                           'goals_for', 'goals_against', 'goal_difference', 'points', 'form'])
                    writer.writeheader()
                    writer.writerows(standings)
                print(f"✅ Standings exported to: {csv_file}")
            except Exception as e:
                print(f"❌ Error exporting CSV: {e}")
    else:
        print("Failed to fetch standings data")
        sys.exit(1)
