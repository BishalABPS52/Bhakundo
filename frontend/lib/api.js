import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const ADMIN_USERNAME = 'bishaladmin';
const ADMIN_PASSWORD = 'plbishal3268';

// Helper to get headers with Basic Auth
const getApiHeaders = () => ({
  'Content-Type': 'application/json',
  'Authorization': 'Basic ' + btoa(`${ADMIN_USERNAME}:${ADMIN_PASSWORD}`)
});

export const predictMatch = async (homeTeam, awayTeam, homeFormation = null, awayFormation = null) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/predict`, {
      home_team: homeTeam,
      away_team: awayTeam,
      home_formation: homeFormation || 'Auto-Predict',
      away_formation: awayFormation || 'Auto-Predict'
    }, {
      headers: getApiHeaders()
    });
    return response.data;
  } catch (error) {
    console.error('Prediction API error:', error);
    throw error;
  }
};

export const getTeams = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/teams`, {
      headers: getApiHeaders()
    });
    return response.data.teams;
  } catch (error) {
    console.error('Teams API error:', error);
    return [
      'AFC Bournemouth',
      'Arsenal FC',
      'Aston Villa FC',
      'Brentford FC',
      'Brighton & Hove Albion FC',
      'Chelsea FC',
      'Crystal Palace FC',
      'Everton FC',
      'Fulham FC',
      'Ipswich Town FC',
      'Leicester City FC',
      'Liverpool FC',
      'Manchester City FC',
      'Manchester United FC',
      'Newcastle United FC',
      'Nottingham Forest FC',
      'Southampton FC',
      'Tottenham Hotspur FC',
      'West Ham United FC',
      'Wolverhampton Wanderers FC'
    ];
  }
};

export const getTeamStats = async (teamName) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/team-stats/${encodeURIComponent(teamName)}`, {
      headers: getApiHeaders()
    });
    return response.data;
  } catch (error) {
    console.error('Team stats API error:', error);
    return null;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/model-info`, {
      headers: getApiHeaders()
    });
    return response.data;
  } catch (error) {
    console.error('Model info API error:', error);
    return {
      base_model: { accuracy: '47.5%', features: 86 },
      lineup_model: { accuracy: '47.5%', features: 97 }
    };
  }
};
