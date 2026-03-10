'use client';

import React, { useState, useEffect } from 'react';
import { Target, Calendar, Clock, CheckCircle, TrendingUp, Brain, ChevronLeft, ChevronRight, Edit2 } from 'lucide-react';
import Image from 'next/image';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://bhakundo-backend.onrender.com';
const ADMIN_USERNAME = 'bishaladmin';
const ADMIN_PASSWORD = 'plbishal3268';

// Helper function to get headers with Basic Auth
const getApiHeaders = () => ({
  'Content-Type': 'application/json',
  'Authorization': 'Basic ' + btoa(`${ADMIN_USERNAME}:${ADMIN_PASSWORD}`)
});

const getTeamLogo = (teamName) => {
  const filename = teamName.replace(/\s+/g, '_').replace('&', 'and') + '.png';
  return `/team-logos/${filename}`;
};

const formatDate = (dateString) => {
  const date = new Date(dateString);
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'];
  
  const dayName = days[date.getDay()];
  const monthName = months[date.getMonth()];
  const day = date.getDate();
  const year = date.getFullYear();
  
  return `${dayName} ${monthName} ${day}, ${year}`;
};

const FormIndicator = ({ form, darkMode }) => {
  if (!form || form.length === 0) return null;
  
  return (
    <div className="flex space-x-1 mt-1">
      {form.map((result, idx) => (
        <div
          key={idx}
          className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white ${
            result === 'W' ? 'bg-green-500' :
            result === 'D' ? 'bg-yellow-500' :
            'bg-red-500'
          }`}
          title={result === 'W' ? 'Win' : result === 'D' ? 'Draw' : 'Loss'}
        >
          {result}
        </div>
      ))}
    </div>
  );
};

export default function Predictor({ darkMode }) {
  const [activeTab, setActiveTab] = useState('predictions');
  const [gameweek, setGameweek] = useState(null);
  const [fixtures, setFixtures] = useState([]);
  const [results, setResults] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [hiddenPredictions, setHiddenPredictions] = useState({});
  const [loading, setLoading] = useState(false);
  const [predictingMatch, setPredictingMatch] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [editingFormations, setEditingFormations] = useState({});
  const [formations, setFormations] = useState({});
  const [isMounted, setIsMounted] = useState(false);
  const [upcomingGW, setUpcomingGW] = useState(null);       // next GW (for Predictions tab)
  const [lastCompletedGW, setLastCompletedGW] = useState(null); // last finished GW (for Results tab)
  const [predictionFilter, setPredictionFilter] = useState('all'); // 'all', 'home_win', 'away_win', 'draw'
  const [accuracyFilter, setAccuracyFilter] = useState('all'); // 'all', 'correct', 'wrong'

  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';
  const accentColor = '#3B82F6';

  // Set mounted state and auto-detect upcoming/last-completed gameweeks.
  // Re-polls every 5 minutes so when a GW completes the UI auto-advances.
  useEffect(() => {
    setIsMounted(true);

    const detectGameweeks = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/smart-gameweek`, {
          headers: getApiHeaders()
        });
        const data = await res.json();
        const upcoming = data.upcoming_gameweek;
        const lastDone = data.last_completed_gameweek;

        setUpcomingGW(prev => {
          // If upcoming GW advanced, update fixtures tab to new upcoming GW
          if (prev !== null && upcoming !== prev) {
            setGameweek(gw => (gw === prev ? upcoming : gw));
          }
          return upcoming;
        });

        setLastCompletedGW(prev => {
          // If last completed GW advanced, update results tab to new last completed GW
          if (prev !== null && lastDone !== prev) {
            setGameweek(gw => (gw === prev ? lastDone : gw));
          }
          return lastDone;
        });

        // On first load, set the gameweek based on current tab
        setGameweek(gw => (gw === null ? (activeTab === 'results' ? lastDone : upcoming) : gw));
      } catch {
        setUpcomingGW(v => v ?? 30);
        setLastCompletedGW(v => v ?? 29);
        setGameweek(gw => gw ?? (activeTab === 'results' ? 29 : 30));
      }
    };

    detectGameweeks();
    // Re-poll every 5 minutes
    const gwInterval = setInterval(detectGameweeks, 5 * 60 * 1000);
    return () => clearInterval(gwInterval);
  }, []);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    if (gameweek !== null) {
      const interval = setInterval(() => {
        if (activeTab === 'predictions') {
          fetchFixtures();
        } else {
          fetchResults();
        }
      }, 60000); // Update every 60 seconds

      if (activeTab === 'predictions') {
        fetchFixtures();
      } else {
        fetchResults();
      }
      return () => clearInterval(interval);
    }
  }, [activeTab, gameweek]);

  const fetchFixtures = async () => {
    if (gameweek === null) return;
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/fixtures?gameweek=${gameweek}`, {
        headers: getApiHeaders()
      });
      const data = await response.json();
      setFixtures(data.fixtures || []);
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Failed to load fixtures:', err);
      setFixtures([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchResults = async () => {
    if (gameweek === null) return;
    try {
      setLoading(true);
      
      // Automatically sync results with predictions in background
      try {
        await fetch(`${API_BASE_URL}/update-prediction-results?gameweek=${gameweek}`, {
          method: 'POST',
          headers: getApiHeaders()
        });
        // Silently sync - no alerts or errors shown to user
      } catch (syncErr) {
        console.log('Background sync skipped:', syncErr);
      }
      
      const response = await fetch(`${API_BASE_URL}/results?gameweek=${gameweek}`, {
        headers: getApiHeaders()
      });
      const data = await response.json();
      setResults(data.results || []);
      
      // Fetch all saved predictions for this gameweek at once
      if (data.results && data.results.length > 0) {
        try {
          const gwPredResponse = await fetch(`${API_BASE_URL}/gameweek-predictions/${gameweek}`, {
            headers: getApiHeaders()
          });
          if (gwPredResponse.ok) {
            const gwPredData = await gwPredResponse.json();
            setPredictions(gwPredData.predictions || {});
          }
        } catch (err) {
          console.error(`Failed to fetch gameweek predictions:`, err);
          // Fall back to individual fetching if bulk fetch fails
          for (const match of data.results) {
            const matchId = match.match_id || String(match.id);
            if (!predictions[matchId]) {
              try {
                const predResponse = await fetch(`${API_BASE_URL}/saved-prediction/${matchId}`, {
                  headers: getApiHeaders()
                });
                if (predResponse.ok) {
                  const predData = await predResponse.json();
                  setPredictions(prev => ({
                    ...prev,
                    [matchId]: predData.prediction
                  }));
                }
                // Don't auto-generate if not found - user must click "Show Predictions" button
              } catch (err) {
                console.error(`Failed to fetch prediction for match ${matchId}:`, err);
              }
            }
          }
        }
      }
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Failed to load results:', err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (match) => {
    const matchId = match.match_id || String(match.id);
    const isCompleted = match.status === 'FINISHED' || match.status === 'COMPLETE';
    setPredictingMatch(matchId);

    try {
      // For finished matches, try to load saved prediction first
      if (isCompleted) {
        try {
          const savedResponse = await fetch(`${API_BASE_URL}/saved-prediction/${matchId}`, {
            headers: getApiHeaders()
          });
          if (savedResponse.ok) {
            const savedData = await savedResponse.json();
            setPredictions(prev => ({
              ...prev,
              [matchId]: savedData.prediction
            }));
            setPredictingMatch(null);
            return;
          }
        } catch (err) {
          console.log('No saved prediction found for match', matchId);
        }
      }

      const matchFormations = formations[matchId] || {
        home_formation: '4-3-3',
        away_formation: '4-3-3'
      };

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify({
          home_team: match.home_team,
          away_team: match.away_team,
          gameweek: match.gameweek,
          match_id: String(matchId),  // Convert to string for backend
          home_formation: matchFormations.home_formation,
          away_formation: matchFormations.away_formation
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error response' }));
        console.error('Prediction API error:', errorData);
        const errorMsg = typeof errorData.detail === 'string' 
          ? errorData.detail 
          : JSON.stringify(errorData.detail || errorData || 'Unknown error');
        alert(`Prediction failed: ${errorMsg}`);
        return;
      }

      const data = await response.json();
      
      // Validate prediction data structure
      if (!data.predicted_score) {
        console.error('Invalid prediction response:', data);
        alert('Prediction response missing score data. Check console for details.');
        return;
      }
      
      setPredictions(prev => ({
        ...prev,
        [matchId]: data
      }));
    } catch (err) {
      console.error('Prediction failed:', err);
      alert(`Prediction error: ${err.message || 'Network or server error'}`);
    } finally {
      setPredictingMatch(null);
    }
  };

  const toggleFormationEdit = (matchId) => {
    setEditingFormations(prev => ({
      ...prev,
      [matchId]: !prev[matchId]
    }));
    
    // Initialize formations if not exists
    if (!formations[matchId]) {
      setFormations(prev => ({
        ...prev,
        [matchId]: {
          home_formation: '4-3-3',
          away_formation: '4-3-3'
        }
      }));
    }
  };

  const updateFormation = (matchId, team, formation) => {
    setFormations(prev => ({
      ...prev,
      [matchId]: {
        ...prev[matchId],
        [team]: formation
      }
    }));
  };

  const getModelPrediction = (probabilities, match) => {
    const homeWin = probabilities.home_win * 100;
    const draw = probabilities.draw * 100;
    const awayWin = probabilities.away_win * 100;
    
    const maxProb = Math.max(homeWin, draw, awayWin);
    
    if (homeWin === maxProb) {
      return { outcome: `${match.home_team} Win`, percentage: homeWin.toFixed(0) };
    } else if (awayWin === maxProb) {
      return { outcome: `${match.away_team} Win`, percentage: awayWin.toFixed(0) };
    } else {
      return { outcome: 'Draw', percentage: draw.toFixed(0) };
    }
  };

  const getPredictionAccuracy = (prediction, match) => {
    if (!prediction || match.home_score === null || match.home_score === undefined) return null;

    const predictedScore = prediction.predicted_score;
    const actualHomeScore = match.home_score;
    const actualAwayScore = match.away_score;

    // Determine actual outcome
    let actualOutcome;
    if (actualHomeScore > actualAwayScore) actualOutcome = 'home_win';
    else if (actualHomeScore < actualAwayScore) actualOutcome = 'away_win';
    else actualOutcome = 'draw';

    // Get predicted outcome from ensemble (final prediction)
    const ensembleProbs = prediction.outcome_probabilities || prediction.base_outcome_probabilities;
    let predictedOutcome;
    const maxProb = Math.max(ensembleProbs.home_win, ensembleProbs.draw, ensembleProbs.away_win);
    if (ensembleProbs.home_win === maxProb) predictedOutcome = 'home_win';
    else if (ensembleProbs.away_win === maxProb) predictedOutcome = 'away_win';
    else predictedOutcome = 'draw';

    return {
      correct: actualOutcome === predictedOutcome,
      predictedOutcome,
      actualOutcome,
      scoreMatch: predictedScore.home === actualHomeScore && predictedScore.away === actualAwayScore
    };
  };

  const calculateGameweekAccuracy = () => {
    if (!results || results.length === 0) return null;

    let totalMatches = 0;
    let correctOutcomes = 0;
    let correctScores = 0;

    results.forEach(match => {
      const matchId = match.match_id || String(match.id);
      const prediction = predictions[matchId];
      const accuracy = prediction ? getPredictionAccuracy(prediction, match) : null;
      
      if (accuracy) {
        totalMatches++;
        if (accuracy.correct) {
          correctOutcomes++;
        }
        if (accuracy.scoreMatch) {
          correctScores++;
        }
      }
    });

    if (totalMatches === 0) return null;

    return {
      total: totalMatches,
      correctOutcomes: correctOutcomes,
      correctScores: correctScores,
      outcomePercentage: ((correctOutcomes / totalMatches) * 100).toFixed(1),
      scorePercentage: ((correctScores / totalMatches) * 100).toFixed(1)
    };
  };

  const handlePrevGameweek = () => {
    if (gameweek > 1) setGameweek(gameweek - 1);
  };

  const handleNextGameweek = () => {
    if (gameweek < 38) setGameweek(gameweek + 1);
  };

  const renderPredictionFixture = (match, index) => {
    const matchId = match.match_id || String(match.id);
    const prediction = predictions[matchId];
    const isPredicting = predictingMatch === matchId;
    const isCompleted = match.status === 'FINISHED' || match.status === 'COMPLETE';
    const hasScore = match.home_score !== null && match.home_score !== undefined;
    const isEditingFormation = editingFormations[matchId];
    const matchFormations = formations[matchId] || { home_formation: '4-3-3', away_formation: '4-3-3' };

    const commonFormations = [
      '3-4-3',
      '3-4-2-1',
      '3-4-1-2',
      '3-5-2',
      '3-5-1-1',
      '4-1-2-1-2',
      '4-1-3-2',
      '4-1-4-1',
      '4-2-2-2',
      '4-2-3-1',
      '4-2-4-0',
      '4-3-1-2',
      '4-3-2-1',
      '4-3-3',
      '4-4-1-1',
      '4-4-2',
      '4-5-1',
      '4-2-1-3',
      '4-3-3 (Defensive)',
      '4-3-3 (Attacking)',
      '5-2-3',
      '5-2-2-1',
      '5-3-2',
      '5-4-1'
    ];

    return (
      <div key={index} className={`${cardBg} border rounded-lg md:rounded-xl p-3 sm:p-4 md:p-6 hover:shadow-lg transition-all`}>
        {/* Header: Date/Time and Gameweek */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3 sm:mb-4 gap-2">
          <div className="flex items-center space-x-2 text-xs sm:text-sm" style={{color: accentColor}}>
            <Calendar size={14} className="sm:w-4 sm:h-4" />
            <span>{formatDate(match.date)}</span>
            <Clock size={14} className="ml-2 sm:w-4 sm:h-4" />
            <span>{new Date(match.date).toLocaleTimeString([], {hour: 'numeric', minute:'2-digit', hour12: true})}</span>
          </div>
          <span className={`px-2 sm:px-3 py-1 rounded-full text-xs font-semibold ${textSecondary}`}>
            GW {match.gameweek}
          </span>
        </div>

        {/* Teams and Score - Mobile Optimized */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4 mb-3 sm:mb-4">
          {/* Home Team */}
          <div className="flex items-center space-x-2 sm:space-x-3 w-full sm:flex-1">
            <Image 
              src={getTeamLogo(match.home_team)} 
              alt={match.home_team}
              width={64}
              height={64}
              className="rounded-lg w-10 h-10 sm:w-12 sm:h-12 md:w-16 md:h-16 flex-shrink-0"
              style={{ width: 'auto', height: 'auto' }}
            />
            <div className="text-left flex-1 min-w-0">
              <p className={`font-bold ${textPrimary} text-sm sm:text-base md:text-lg truncate`}>{match.home_team}</p>
              <p className={`text-xs ${textSecondary}`}>Home</p>
              <div className="hidden sm:block"><FormIndicator form={match.home_form} darkMode={darkMode} /></div>
            </div>
          </div>

          {/* Score/VS - Centered */}
          <div className="px-2 sm:px-4 md:px-6 flex-shrink-0">
            {hasScore ? (
              <div className="text-center">
                <p className={`text-xs ${textSecondary} mb-1`}>Actual Score</p>
                <div className="flex items-center justify-center space-x-2 sm:space-x-3 md:space-x-4">
                  <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${textPrimary}`}>{match.home_score}</span>
                  <span className={`text-lg sm:text-xl md:text-2xl ${textSecondary}`}>-</span>
                  <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${textPrimary}`}>{match.away_score}</span>
                </div>
              </div>
            ) : (
              <span className={`text-xl sm:text-2xl font-bold ${textSecondary}`}>VS</span>
            )}
          </div>

          {/* Away Team */}
          <div className="flex items-center space-x-2 sm:space-x-3 flex-row-reverse w-full sm:flex-1">
            <Image 
              src={getTeamLogo(match.away_team)} 
              alt={match.away_team}
              width={64}
              height={64}
              className="rounded-lg w-10 h-10 sm:w-12 sm:h-12 md:w-16 md:h-16 flex-shrink-0"
              style={{ width: 'auto', height: 'auto' }}
            />
            <div className="text-right flex-1 min-w-0">
              <p className={`font-bold ${textPrimary} text-sm sm:text-base md:text-lg truncate`}>{match.away_team}</p>
              <p className={`text-xs ${textSecondary}`}>Away</p>
              <div className="hidden sm:flex justify-end"><FormIndicator form={match.away_form} darkMode={darkMode} /></div>
            </div>
          </div>
        </div>

        {/* Form Indicators for Mobile - Below teams */}
        <div className="flex sm:hidden items-center justify-between mb-3 px-2">
          <div className="flex-1">
            <FormIndicator form={match.home_form} darkMode={darkMode} />
          </div>
          <div className="flex-1 flex justify-end">
            <FormIndicator form={match.away_form} darkMode={darkMode} />
          </div>
        </div>

        {/* Formation Editor */}
        <div className="mb-3 sm:mb-4">
          <button
            onClick={() => toggleFormationEdit(matchId)}
            className={`w-full py-2 px-3 sm:px-4 rounded-lg flex items-center justify-center space-x-2 ${darkMode ? 'bg-slate-700 hover:bg-slate-600' : 'bg-gray-100 hover:bg-gray-200'} transition-all`}
          >
            <Edit2 size={14} className="sm:w-4 sm:h-4" />
            <span className={`text-xs sm:text-sm font-semibold ${textSecondary}`}>
              {isEditingFormation ? 'Hide Formations' : 'Edit Formations'}
            </span>
          </button>

          {isEditingFormation && (
            <div className={`mt-3 p-3 sm:p-4 rounded-lg ${darkMode ? 'bg-slate-700/50' : 'bg-gray-50'} space-y-3`}>
              {/* Home Formation */}
              <div>
                <label className={`block text-xs ${textSecondary} mb-2`}>
                  {match.home_team} Formation
                </label>
                <select
                  value={matchFormations.home_formation}
                  onChange={(e) => updateFormation(matchId, 'home_formation', e.target.value)}
                  className={`w-full p-2 text-sm rounded ${darkMode ? 'bg-slate-600 text-white' : 'bg-white text-slate-900'} border ${darkMode ? 'border-slate-500' : 'border-gray-300'}`}
                >
                  {commonFormations.map(formation => (
                    <option key={formation} value={formation}>{formation}</option>
                  ))}
                </select>
              </div>

              {/* Away Formation */}
              <div>
                <label className={`block text-xs ${textSecondary} mb-2`}>
                  {match.away_team} Formation
                </label>
                <select
                  value={matchFormations.away_formation}
                  onChange={(e) => updateFormation(matchId, 'away_formation', e.target.value)}
                  className={`w-full p-2 text-sm rounded ${darkMode ? 'bg-slate-600 text-white' : 'bg-white text-slate-900'} border ${darkMode ? 'border-slate-500' : 'border-gray-300'}`}
                >
                  {commonFormations.map(formation => (
                    <option key={formation} value={formation}>{formation}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Predict Button or Prediction Results */}
        {!prediction ? (
          <button
            onClick={() => handlePredict(match)}
            disabled={isPredicting}
            className="w-full mt-3 sm:mt-4 py-2.5 sm:py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg text-sm sm:text-base font-semibold hover:shadow-lg transition-all disabled:opacity-50 flex items-center justify-center space-x-2"
          >
            {isPredicting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 sm:h-5 sm:w-5 border-b-2 border-white"></div>
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <Brain size={18} className="sm:w-5 sm:h-5" />
                <span>Predict Match</span>
              </>
            )}
          </button>
        ) : (
          <div className={`mt-3 sm:mt-4 p-3 sm:p-4 rounded-lg ${darkMode ? 'bg-slate-700/50' : 'bg-blue-50'}`}>

            {/* Bhakundo Verdict - Mobile Optimized */}
            <div className="mb-3 sm:mb-4 p-2 sm:p-3 rounded-lg bg-teal-500/20 border border-teal-500/50">
              <p className={`text-xs sm:text-sm font-semibold ${textPrimary} text-center`}>
                <span className="hidden sm:inline">Bhakundo Predicts: </span>{' '}
                <span className={
                  prediction.outcome_probabilities.home_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.away_win) ? 'text-green-400' :
                  prediction.outcome_probabilities.away_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.home_win) ? 'text-teal-400' :
                  'text-yellow-400'
                }>
                  {prediction.outcome_probabilities.home_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.away_win) ? `${match.home_team} Win` :
                   prediction.outcome_probabilities.away_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.home_win) ? `${match.away_team} Win` :
                   'Draw'}
                </span>
              </p>
            </div>

            {/* Predicted Score - Mobile Optimized */}
            <div className="flex items-center justify-center space-x-4 sm:space-x-6 mb-3 sm:mb-4">
              <div className="text-center">
                <p className={`text-xs ${textSecondary} mb-1`}>Predicted Score</p>
                <div className={`flex items-center space-x-2 sm:space-x-3 p-2 sm:p-3 rounded-lg ${
                  (prediction.predicted_score?.home ?? 0) > (prediction.predicted_score?.away ?? 0) ? 'bg-green-500/20' :
                  (prediction.predicted_score?.home ?? 0) === (prediction.predicted_score?.away ?? 0) ? 'bg-yellow-500/20' :
                  'bg-teal-500/20'
                }`}>
                  <span className={`text-2xl sm:text-3xl font-bold ${
                    (prediction.predicted_score?.home ?? 0) > (prediction.predicted_score?.away ?? 0) ? 'text-green-400' :
                    (prediction.predicted_score?.home ?? 0) === (prediction.predicted_score?.away ?? 0) ? 'text-yellow-400' :
                    textPrimary
                  }`}>
                    {prediction.predicted_score?.home ?? 1}
                  </span>
                  <span className={`text-lg sm:text-xl ${textSecondary}`}>-</span>
                  <span className={`text-2xl sm:text-3xl font-bold ${
                    (prediction.predicted_score?.away ?? 0) > (prediction.predicted_score?.home ?? 0) ? 'text-teal-400' :
                    (prediction.predicted_score?.home ?? 0) === (prediction.predicted_score?.away ?? 0) ? 'text-yellow-400' :
                    textPrimary
                  }`}>
                    {prediction.predicted_score?.away ?? 1}
                  </span>
                </div>
              </div>
            </div>

            {/* Model Predictions - Mobile Optimized Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mb-3 sm:mb-4">
              {/* Base Model */}
              <div className={`p-2 sm:p-3 rounded-lg ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                <p className={`text-xs ${textSecondary} mb-2 flex items-center justify-between`}>
                  <span className="font-semibold">Base Model</span>
                  <Brain size={12} className="sm:w-3.5 sm:h-3.5" />
                </p>
                <div className="space-y-1 mb-2">
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Home Win</span>
                    <span className="text-green-400 font-semibold">
                      {(prediction.base_outcome_probabilities.home_win * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Draw</span>
                    <span className="text-yellow-400 font-semibold">
                      {(prediction.base_outcome_probabilities.draw * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Away Win</span>
                    <span className="text-teal-400 font-semibold">
                      {(prediction.base_outcome_probabilities.away_win * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className={`text-xs p-1.5 sm:p-2 rounded ${darkMode ? 'bg-slate-700' : 'bg-blue-100'} text-center`}>
                  <p className={`font-bold ${textPrimary} truncate`}>
                    <span className={
                      prediction.base_outcome_probabilities.home_win > Math.max(prediction.base_outcome_probabilities.draw, prediction.base_outcome_probabilities.away_win) ? 'text-green-400' :
                      prediction.base_outcome_probabilities.draw > prediction.base_outcome_probabilities.away_win ? 'text-yellow-400' :
                      'text-teal-400'
                    }>
                      {prediction.base_outcome_probabilities.home_win > Math.max(prediction.base_outcome_probabilities.draw, prediction.base_outcome_probabilities.away_win) ? `${match.home_team} Win` :
                       prediction.base_outcome_probabilities.draw > prediction.base_outcome_probabilities.away_win ? 'Draw' :
                       `${match.away_team} Win`}
                    </span>
                  </p>
                </div>
              </div>

              {/* Lineup Model */}
              <div className={`p-2 sm:p-3 rounded-lg ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                <p className={`text-xs ${textSecondary} mb-2 flex items-center justify-between`}>
                  <span className="font-semibold">Lineup Model</span>
                  <TrendingUp size={12} className="sm:w-3.5 sm:h-3.5" />
                </p>
                <div className="space-y-1 mb-2">
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Home Win</span>
                    <span className="text-green-400 font-semibold">
                      {(prediction.lineup_outcome_probabilities.home_win * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Draw</span>
                    <span className="text-yellow-400 font-semibold">
                      {(prediction.lineup_outcome_probabilities.draw * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Away Win</span>
                    <span className="text-teal-400 font-semibold">
                      {(prediction.lineup_outcome_probabilities.away_win * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className={`text-xs p-1.5 sm:p-2 rounded ${darkMode ? 'bg-slate-700' : 'bg-blue-100'} text-center`}>
                  <p className={`font-bold ${textPrimary} truncate`}>
                    <span className={
                      prediction.lineup_outcome_probabilities.home_win > Math.max(prediction.lineup_outcome_probabilities.draw, prediction.lineup_outcome_probabilities.away_win) ? 'text-green-400' :
                      prediction.lineup_outcome_probabilities.draw > prediction.lineup_outcome_probabilities.away_win ? 'text-yellow-400' :
                      'text-teal-400'
                    }>
                      {prediction.lineup_outcome_probabilities.home_win > Math.max(prediction.lineup_outcome_probabilities.draw, prediction.lineup_outcome_probabilities.away_win) ? `${match.home_team} Win` :
                       prediction.lineup_outcome_probabilities.draw > prediction.lineup_outcome_probabilities.away_win ? 'Draw' :
                       `${match.away_team} Win`}
                    </span>
                  </p>
                </div>
              </div>
            </div>

            {/* Real Score Status */}
            {!isCompleted && (
              <div className={`text-center py-2 px-3 sm:px-4 rounded ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                <p className={`text-xs sm:text-sm ${textSecondary}`}>⏳ Waiting for match to complete...</p>
              </div>
            )}
          </div>
        )}

        {/* Stadium */}
        {match.venue && (
          <div className={`mt-2 sm:mt-3 text-xs sm:text-sm ${textSecondary} text-center truncate`}>
            Stadium: {match.venue}
          </div>
        )}
      </div>
    );
  };

  const renderResult = (match, index) => {
    const matchId = match.match_id || String(match.id);
    const prediction = predictions[matchId];
    const isCompleted = match.status === 'FINISHED' || match.status === 'COMPLETE';
    const hasScore = match.home_score !== null && match.home_score !== undefined;
    const accuracy = prediction ? getPredictionAccuracy(prediction, match) : null;
    const isPredictionHidden = hiddenPredictions[matchId];

    return (
      <div key={index} className={`${cardBg} border rounded-lg md:rounded-xl p-3 sm:p-4 md:p-6 hover:shadow-lg transition-all ${accuracy ? (accuracy.correct ? 'ring-2 ring-green-500' : 'ring-2 ring-red-500') : ''}`}>
        {/* Header: Date/Time and Status */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3 sm:mb-4 gap-2">
          <div className="flex items-center space-x-2 text-xs sm:text-sm" style={{color: accentColor}}>
            <Calendar size={14} className="sm:w-4 sm:h-4" />
            <span>{formatDate(match.date)}</span>
            {isCompleted && (
              <>
                <CheckCircle size={14} className="ml-2 text-green-500 sm:w-4 sm:h-4" />
                <span className="text-green-500 text-xs sm:text-sm font-semibold">Full Time</span>
              </>
            )}
          </div>
          <span className={`px-2 sm:px-3 py-1 rounded-full text-xs font-semibold ${textSecondary}`}>
            GW {match.gameweek}
          </span>
        </div>

        {/* Teams and Actual Score - Mobile Optimized */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4 mb-3 sm:mb-4">
          {/* Home Team */}
          <div className="flex items-center space-x-2 sm:space-x-3 w-full sm:flex-1">
            <Image 
              src={getTeamLogo(match.home_team)} 
              alt={match.home_team}
              width={64}
              height={64}
              className="rounded-lg w-10 h-10 sm:w-12 sm:h-12 md:w-16 md:h-16 flex-shrink-0"
              style={{ width: 'auto', height: 'auto' }}
            />
            <div className="text-left flex-1 min-w-0">
              <p className={`font-bold ${textPrimary} text-sm sm:text-base md:text-lg truncate`}>{match.home_team}</p>
              <p className={`text-xs ${textSecondary}`}>Home</p>
            </div>
          </div>

          {/* Real Score */}
          <div className="px-2 sm:px-4 md:px-6 flex-shrink-0">
            {hasScore ? (
              <div className="text-center">
                <p className={`text-xs ${textSecondary} mb-1`}>Actual Score</p>
                <div className="flex items-center justify-center space-x-2 sm:space-x-3 md:space-x-4 mb-2">
                  <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${textPrimary}`}>{match.home_score}</span>
                  <span className={`text-lg sm:text-xl md:text-2xl ${textSecondary}`}>-</span>
                  <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${textPrimary}`}>{match.away_score}</span>
                </div>
                {match.venue && (
                  <p className={`text-xs ${textSecondary}`}>
                    Stadium: {match.venue}
                  </p>
                )}
              </div>
            ) : (
              <span className={`text-xl sm:text-2xl font-bold ${textSecondary}`}>VS</span>
            )}
          </div>

          {/* Away Team */}
          <div className="flex items-center space-x-2 sm:space-x-3 flex-row-reverse w-full sm:flex-1">
            <Image 
              src={getTeamLogo(match.away_team)} 
              alt={match.away_team}
              width={64}
              height={64}
              className="rounded-lg w-10 h-10 sm:w-12 sm:h-12 md:w-16 md:h-16 flex-shrink-0"
              style={{ width: 'auto', height: 'auto' }}
            />
            <div className="text-right flex-1 min-w-0">
              <p className={`font-bold ${textPrimary} text-sm sm:text-base md:text-lg truncate`}>{match.away_team}</p>
              <p className={`text-xs ${textSecondary}`}>Away</p>
            </div>
          </div>
        </div>

        {/* Show Predictions Button - Default hidden until clicked */}
        {prediction && isPredictionHidden !== false && (
          <button
            onClick={() => setHiddenPredictions(prev => ({...prev, [matchId]: false}))}
            className="w-full mt-3 sm:mt-4 py-2.5 sm:py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg text-sm sm:text-base font-semibold hover:shadow-lg transition-all flex items-center justify-center space-x-2"
          >
            <span>Show Predictions</span>
          </button>
        )}

        {/* Loading Button - When prediction is being fetched */}
        {!prediction && predictingMatch === matchId && (
          <button
            disabled
            className="w-full mt-3 sm:mt-4 py-2.5 sm:py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg text-sm sm:text-base font-semibold opacity-50 flex items-center justify-center space-x-2"
          >
            <div className="animate-spin rounded-full h-4 w-4 sm:h-5 sm:w-5 border-b-2 border-white"></div>
            <span>Loading Prediction...</span>
          </button>
        )}

        {/* Prediction Results */}
        {prediction && isPredictionHidden === false && (
          <div className={`mt-3 sm:mt-4 p-3 sm:p-4 rounded-lg ${darkMode ? 'bg-slate-700/50' : 'bg-blue-50'} space-y-3 sm:space-y-4`}>
            {/* Prediction Accuracy */}
            {accuracy && (
              <div className={`text-center py-2 sm:py-3 px-3 sm:px-4 rounded-lg ${accuracy.correct ? 'bg-green-500/20 border-2 border-green-500' : 'bg-red-500/20 border-2 border-red-500'}`}>
                <p className={`text-base sm:text-lg font-bold ${accuracy.correct ? 'text-green-500' : 'text-red-500'}`}>
                  {accuracy.correct ? 'Prediction was Correct' : 'Prediction was Wrong'}
                </p>
                <p className={`text-xs sm:text-sm ${textSecondary} mt-1`}>
                  {accuracy.scoreMatch 
                    ? 'Score matched Exactly' 
                    : accuracy.correct 
                      ? 'Score didn\'t match' 
                      : ''}
                </p>
              </div>
            )}

            {/* Bhakundo Prediction Box */}
            <div className={`p-3 rounded-lg ${accuracy && !accuracy.correct ? 'bg-red-500/20 border border-red-500/50' : 'bg-teal-500/20 border border-teal-500/50'}`}>
              <p className={`text-sm font-semibold ${textPrimary} text-center`}>
                 Bhakundo Predicted: {' '}
                <span className={
                  prediction.outcome_probabilities.home_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.away_win) ? 'text-green-400' :
                  prediction.outcome_probabilities.away_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.home_win) ? 'text-teal-400' :
                  'text-yellow-400'
                }>
                  {prediction.outcome_probabilities.home_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.away_win) ? `${match.home_team} Win` :
                   prediction.outcome_probabilities.away_win > Math.max(prediction.outcome_probabilities.draw, prediction.outcome_probabilities.home_win) ? `${match.away_team} Win` :
                   'Draw'}
                </span>
              </p>
            </div>

            {/* Predicted Score */}
            <div className="flex items-center justify-center space-x-4 sm:space-x-6 mb-3 sm:mb-4">
              <div className="text-center">
                <p className={`text-xs ${textSecondary} mb-1`}>Predicted Score</p>
                <div className={`flex items-center space-x-2 sm:space-x-3 p-2 sm:p-3 rounded-lg ${
                  (prediction.predicted_score?.home ?? 0) > (prediction.predicted_score?.away ?? 0) ? 'bg-green-500/20' :
                  (prediction.predicted_score?.home ?? 0) === (prediction.predicted_score?.away ?? 0) ? 'bg-yellow-500/20' :
                  'bg-teal-500/20'
                }`}>
                  <span className={`text-2xl sm:text-3xl font-bold ${
                    (prediction.predicted_score?.home ?? 0) > (prediction.predicted_score?.away ?? 0) ? 'text-green-400' :
                    (prediction.predicted_score?.home ?? 0) === (prediction.predicted_score?.away ?? 0) ? 'text-yellow-400' :
                    textPrimary
                  }`}>
                    {prediction.predicted_score?.home ?? 1}
                  </span>
                  <span className={`text-lg sm:text-xl ${textSecondary}`}>-</span>
                  <span className={`text-2xl sm:text-3xl font-bold ${
                    (prediction.predicted_score?.away ?? 0) > (prediction.predicted_score?.home ?? 0) ? 'text-teal-400' :
                    (prediction.predicted_score?.home ?? 0) === (prediction.predicted_score?.away ?? 0) ? 'text-yellow-400' :
                    textPrimary
                  }`}>
                    {prediction.predicted_score?.away ?? 1}
                  </span>
                </div>
              </div>
            </div>

            {/* Model Predictions */}
            <div className="grid grid-cols-2 gap-4">
              {/* Base Model */}
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                <p className={`text-xs ${textSecondary} mb-2 flex items-center justify-between`}>
                  <span className="font-semibold">Base Model</span>
                  <Brain size={14} />
                </p>
                <div className="space-y-1 mb-2">
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Home Win</span>
                    <span className="text-green-400 font-semibold">{(prediction.base_outcome_probabilities.home_win * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Draw</span>
                    <span className="text-yellow-400 font-semibold">{(prediction.base_outcome_probabilities.draw * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Away Win</span>
                    <span className="text-teal-400 font-semibold">{(prediction.base_outcome_probabilities.away_win * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className={`text-xs p-2 rounded ${darkMode ? 'bg-slate-700' : 'bg-blue-100'} text-center`}>
                  <p className={`font-bold ${textPrimary}`}>
                    {getModelPrediction(prediction.base_outcome_probabilities, match).outcome}
                  </p>
                </div>
              </div>

              {/* Lineup Model */}
              <div className={`p-3 rounded-lg ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                <p className={`text-xs ${textSecondary} mb-2 flex items-center justify-between`}>
                  <span className="font-semibold">Lineup Model</span>
                  <TrendingUp size={14} />
                </p>
                <div className="space-y-1 mb-2">
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Home Win</span>
                    <span className="text-green-400 font-semibold">{(prediction.lineup_outcome_probabilities.home_win * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Draw</span>
                    <span className="text-yellow-400 font-semibold">{(prediction.lineup_outcome_probabilities.draw * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>Away Win</span>
                    <span className="text-teal-400 font-semibold">{(prediction.lineup_outcome_probabilities.away_win * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className={`text-xs p-2 rounded ${darkMode ? 'bg-slate-700' : 'bg-blue-100'} text-center`}>
                  <p className={`font-bold ${textPrimary}`}>
                    {getModelPrediction(prediction.lineup_outcome_probabilities, match).outcome}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Hide Predictions Button - Below prediction card when shown */}
        {prediction && isPredictionHidden === false && (
          <button
            onClick={() => setHiddenPredictions(prev => ({...prev, [matchId]: true}))}
            className={`w-full mt-3 sm:mt-4 py-2.5 sm:py-3 ${darkMode ? 'bg-slate-600 hover:bg-slate-500' : 'bg-gray-400 hover:bg-gray-500'} text-white rounded-lg text-sm sm:text-base font-semibold hover:shadow-lg transition-all flex items-center justify-center space-x-2`}
          >
            <span>Hide Predictions</span>
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="px-4 sm:px-6 md:px-8 lg:px-12 py-6 md:py-12 w-full overflow-x-hidden">
      <div className="max-w-7xl mx-auto w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className={`text-5xl font-bold ${textPrimary} mb-4`}>Bhakundo Predictor</h1>
          <p className={`text-xl ${textSecondary}`}>Prepare, Predict & Play </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6 md:mb-8">
          <div className="flex items-center gap-2 sm:gap-4 overflow-x-auto pb-2 sm:pb-0">
            <button
              onClick={() => {
                setActiveTab('predictions');
                if (activeTab !== 'predictions' && upcomingGW) {
                  setGameweek(upcomingGW);
                }
              }}
              className={`flex items-center space-x-2 px-4 sm:px-6 py-2.5 sm:py-3 rounded-lg font-semibold transition-all whitespace-nowrap text-sm sm:text-base ${
                activeTab === 'predictions'
                  ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                  : `${cardBg} border ${textSecondary} hover:bg-blue-500/10`
              }`}
            >
              <Target size={18} className="sm:w-5 sm:h-5" />
              <span className="hidden sm:inline">Predictions</span>
              <span className="sm:hidden">Predict</span>
            </button>
            <button
              onClick={() => {
                setActiveTab('results');
                if (activeTab !== 'results' && lastCompletedGW) {
                  setGameweek(lastCompletedGW);
                }
              }}
              className={`flex items-center space-x-2 px-4 sm:px-6 py-2.5 sm:py-3 rounded-lg font-semibold transition-all whitespace-nowrap text-sm sm:text-base ${
                activeTab === 'results'
                  ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                  : `${cardBg} border ${textSecondary} hover:bg-blue-500/10`
              }`}
            >
              <CheckCircle size={18} className="sm:w-5 sm:h-5" />
              <span className="hidden sm:inline">Results & Accuracy</span>
              <span className="sm:hidden">Results</span>
            </button>
          </div>

          {/* Last Updated */}
          <div className={`text-xs sm:text-sm ${textSecondary} flex items-center space-x-2 justify-center sm:justify-start`}>
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Updated: {isMounted ? lastUpdated.toLocaleTimeString() : '--:--:--'}</span>
          </div>
        </div>

        {/* Gameweek Selector */}
        <div className={`${cardBg} border rounded-xl md:rounded-2xl p-4 sm:p-6 mb-6 md:mb-8 shadow-xl`}>
          <div className="flex items-center justify-between gap-4">
            <button
              onClick={handlePrevGameweek}
              disabled={gameweek === null || gameweek <= 1}
              className="p-2 sm:p-3 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft size={20} className="sm:w-6 sm:h-6" />
            </button>

            <div className="text-center flex-1">
              <p className={`text-xs sm:text-sm ${textSecondary} mb-1`}>Gameweek</p>
              <p className={`text-3xl sm:text-4xl font-bold ${textPrimary}`}>{gameweek !== null ? gameweek : '--'}</p>
            </div>

            <button
              onClick={handleNextGameweek}
              disabled={gameweek === null || gameweek >= 38}
              className="p-2 sm:p-3 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight size={20} className="sm:w-6 sm:h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className={`mt-4 ${textSecondary}`}>Loading matches...</p>
          </div>
        ) : (
          <>
            {/* Gameweek Accuracy Card - Only show in Results tab */}
            {activeTab === 'results' && calculateGameweekAccuracy() && (
              <div className={`${cardBg} border rounded-2xl p-6 mb-6 shadow-xl`}>
                <h3 className={`text-lg font-semibold ${textPrimary} mb-4`}>GW {gameweek} Prediction Accuracy</h3>
                
                {/* Outcome Accuracy */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <h4 className={`text-base font-semibold ${textPrimary}`}>GW Prediction Accuracy (Win/Draw/Loss)</h4>
                      <p className={`text-sm ${textSecondary}`}>
                        {calculateGameweekAccuracy().correctOutcomes} correct out of {calculateGameweekAccuracy().total} matches
                      </p>
                    </div>
                    <div className="text-center">
                      <div className={`text-4xl font-bold ${calculateGameweekAccuracy().outcomePercentage >= 50 ? 'text-green-500' : calculateGameweekAccuracy().outcomePercentage >= 33 ? 'text-yellow-500' : 'text-red-500'}`}>
                        {calculateGameweekAccuracy().outcomePercentage}%
                      </div>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all ${calculateGameweekAccuracy().outcomePercentage >= 50 ? 'bg-green-500' : calculateGameweekAccuracy().outcomePercentage >= 33 ? 'bg-yellow-500' : 'bg-red-500'}`}
                      style={{ width: `${calculateGameweekAccuracy().outcomePercentage}%` }}
                    ></div>
                  </div>
                </div>

                {/* Score Accuracy */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <h4 className={`text-base font-semibold ${textPrimary}`}>GW Score Accuracy (Exact Score)</h4>
                      <p className={`text-sm ${textSecondary}`}>
                        {calculateGameweekAccuracy().correctScores} exact scores out of {calculateGameweekAccuracy().total} matches
                      </p>
                    </div>
                    <div className="text-center">
                      <div className={`text-4xl font-bold ${calculateGameweekAccuracy().scorePercentage >= 30 ? 'text-green-500' : calculateGameweekAccuracy().scorePercentage >= 15 ? 'text-yellow-500' : 'text-red-500'}`}>
                        {calculateGameweekAccuracy().scorePercentage}%
                      </div>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all ${calculateGameweekAccuracy().scorePercentage >= 30 ? 'bg-green-500' : calculateGameweekAccuracy().scorePercentage >= 15 ? 'bg-yellow-500' : 'bg-red-500'}`}
                      style={{ width: `${calculateGameweekAccuracy().scorePercentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            )}

            {/* Filters */}
            <div className={`${cardBg} border rounded-xl p-4 mb-6 shadow-md`}>
              {activeTab === 'predictions' ? (
                // Prediction Filter for Predictions Tab
                <>
                  <p className={`text-sm ${textSecondary} mb-3 font-semibold`}>Filter Predictions</p>
                  
                  {/* Mobile Dropdown */}
                  <div className="md:hidden">
                    <select
                      value={predictionFilter}
                      onChange={(e) => setPredictionFilter(e.target.value)}
                      className={`w-full px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                        darkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-300'
                      } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    >
                      <option value="all">All Predictions</option>
                      <option value="home_win">Home Win Predictions</option>
                      <option value="draw">Draw Predictions</option>
                      <option value="away_win">Away Win Predictions</option>
                    </select>
                  </div>

                  {/* Desktop Buttons */}
                  <div className="hidden md:flex flex-wrap gap-2">
                    <button
                      onClick={() => setPredictionFilter('all')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        predictionFilter === 'all'
                          ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-blue-500/20`
                      }`}
                    >
                      All Predictions
                    </button>
                    <button
                      onClick={() => setPredictionFilter('home_win')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        predictionFilter === 'home_win'
                          ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-green-500/20`
                      }`}
                    >
                      Home Win Predictions
                    </button>
                    <button
                      onClick={() => setPredictionFilter('draw')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        predictionFilter === 'draw'
                          ? 'bg-gradient-to-r from-yellow-500 to-amber-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-yellow-500/20`
                      }`}
                    >
                      Draw Predictions
                    </button>
                    <button
                      onClick={() => setPredictionFilter('away_win')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        predictionFilter === 'away_win'
                          ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-teal-500/20`
                      }`}
                    >
                      Away Win Predictions
                    </button>
                  </div>
                </>
              ) : (
                // Accuracy Filter for Results Tab
                <>
                  <p className={`text-sm ${textSecondary} mb-3 font-semibold`}>Filter by Accuracy</p>
                  
                  {/* Mobile Dropdown */}
                  <div className="md:hidden">
                    <select
                      value={accuracyFilter}
                      onChange={(e) => setAccuracyFilter(e.target.value)}
                      className={`w-full px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                        darkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-300'
                      } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    >
                      <option value="all">All Results</option>
                      <option value="correct">Correct Predictions</option>
                      <option value="wrong">Wrong Predictions</option>
                    </select>
                  </div>

                  {/* Desktop Buttons */}
                  <div className="hidden md:flex flex-wrap gap-2">
                    <button
                      onClick={() => setAccuracyFilter('all')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        accuracyFilter === 'all'
                          ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-blue-500/20`
                      }`}
                    >
                      All Results
                    </button>
                    <button
                      onClick={() => setAccuracyFilter('correct')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        accuracyFilter === 'correct'
                          ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-green-500/20`
                      }`}
                    >
                      Correct Predictions
                    </button>
                    <button
                      onClick={() => setAccuracyFilter('wrong')}
                      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                        accuracyFilter === 'wrong'
                          ? 'bg-gradient-to-r from-red-500 to-rose-500 text-white shadow-lg'
                          : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-red-500/20`
                      }`}
                    >
                      Wrong Predictions
                    </button>
                  </div>
                </>
              )}
            </div>

            <div className="grid grid-cols-1 gap-6">
              {activeTab === 'predictions'
                ? (() => {
                    let filteredFixtures = fixtures;
                    
                    if (predictionFilter !== 'all') {
                      filteredFixtures = fixtures.filter(fixture => {
                        const prediction = predictions[fixture.id];
                        if (!prediction || !prediction.outcome_probabilities) return false;
                        
                        const probs = prediction.outcome_probabilities;
                        const maxProb = Math.max(probs.home_win, probs.draw, probs.away_win);
                        
                        if (predictionFilter === 'home_win') return probs.home_win === maxProb;
                        if (predictionFilter === 'draw') return probs.draw === maxProb;
                        if (predictionFilter === 'away_win') return probs.away_win === maxProb;
                        return false;
                      });
                    }
                    
                    return filteredFixtures.length > 0 
                      ? filteredFixtures.map((fixture, index) => renderPredictionFixture(fixture, index))
                      : <div className={`${cardBg} border rounded-xl p-12 text-center`}>
                          <p className={`text-lg ${textSecondary}`}>No matches found for the selected filter</p>
                        </div>;
                  })()
                : (() => {
                    let filteredResults = results;
                    
                    if (accuracyFilter !== 'all') {
                      filteredResults = results.filter(result => {
                        const prediction = predictions[result.match_id || String(result.id)];
                        const accuracy = prediction ? getPredictionAccuracy(prediction, result) : null;
                        
                        if (!accuracy) return false;
                        if (accuracyFilter === 'correct') return accuracy.correct;
                        if (accuracyFilter === 'wrong') return !accuracy.correct;
                        return false;
                      });
                    }
                    
                    return filteredResults.length > 0
                      ? filteredResults.map((result, index) => renderResult(result, index))
                      : <div className={`${cardBg} border rounded-xl p-12 text-center`}>
                          <p className={`text-lg ${textSecondary}`}>No matches found for the selected filter</p>
                        </div>;
                  })()
              }
            </div>
          </>
        )}
      </div>
    </div>
  );
}
