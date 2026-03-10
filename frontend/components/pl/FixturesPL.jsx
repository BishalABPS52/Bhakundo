'use client';

import React, { useState, useEffect } from 'react';
import { Calendar, Clock, CheckCircle, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import Image from 'next/image';

// API Configuration — reads NEXT_PUBLIC_API_URL env var (set in .env.local for local dev)
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

export default function Fixtures({ darkMode }) {
  const [gameweek, setGameweek] = useState(null);
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [isMounted, setIsMounted] = useState(false);
  const [resultFilter, setResultFilter] = useState('all'); // 'all', 'home_win', 'away_win', 'draw'

  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';
  const accentColor = '#3B82F6';

  useEffect(() => {
    setIsMounted(true);
    // Auto-detect upcoming gameweek; re-poll every 5 minutes
    const detectGameweek = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/smart-gameweek`, {
          headers: getApiHeaders()
        });
        const data = await res.json();
        const upcoming = data.upcoming_gameweek || data.fixtures_gameweek || 30;
        setGameweek(prev => (prev === null ? upcoming : prev));
      } catch {
        setGameweek(prev => (prev === null ? 30 : prev));
      }
    };
    detectGameweek();
    const gwPoll = setInterval(detectGameweek, 5 * 60 * 1000);
    return () => clearInterval(gwPoll);
  }, []);

  useEffect(() => {
    if (gameweek !== null) {
      const interval = setInterval(() => {
        fetchMatches();
      }, 300000); // Update every 5 minutes (300000 ms)
      fetchMatches();
      return () => clearInterval(interval);
    }
  }, [gameweek]);

  const fetchMatches = async () => {
    if (gameweek === null) return;
    try {
      setLoading(true);
      
      // Always use /fixtures endpoint — backend handles past gameweeks transparently
      const endpoint = `${API_BASE_URL}/fixtures?gameweek=${gameweek}`;
      
      const response = await fetch(endpoint, {
        headers: getApiHeaders()
      });
      const data = await response.json();
      
      // Handle both results and fixtures response formats
      const matchesData = data.results || data.fixtures || [];
      setMatches(matchesData);
      setMessage(data.message || '');
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Failed to load matches:', err);
      setMatches([]);
    } finally {
      setLoading(false);
    }
  };

  const handlePrevGameweek = () => {
    if (gameweek > 1) setGameweek(gameweek - 1);
  };

  const handleNextGameweek = () => {
    if (gameweek < 38) setGameweek(gameweek + 1);
  };

  const renderMatch = (match, index) => {
    const isLive = match.status === 'LIVE' || match.status === 'IN_PLAY';
    const isCompleted = match.status === 'FINISHED' || match.status === 'COMPLETE';
    // Handle both score field names from backend (home_score/home_goals)
    const homeScore = match.home_score !== undefined ? match.home_score : match.home_goals;
    const awayScore = match.away_score !== undefined ? match.away_score : match.away_goals;
    const hasScore = homeScore !== null && homeScore !== undefined;
    
    return (
      <div key={index} className={`${cardBg} border rounded-lg md:rounded-xl p-3 sm:p-4 md:p-6 hover:shadow-lg transition-all ${isLive ? 'ring-2 ring-red-500 animate-pulse' : ''}`}>
        {/* Header: Date/Time and Status Badges */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3 sm:mb-4 gap-2">
          <div className="flex items-center space-x-2 text-xs sm:text-sm" style={{color: accentColor}}>
            <Calendar size={14} className="sm:w-4 sm:h-4" />
            <span>{formatDate(match.date)}</span>
            <Clock size={14} className="ml-2 sm:w-4 sm:h-4" />
            <span>{new Date(match.date).toLocaleTimeString([], {hour: 'numeric', minute:'2-digit', hour12: true})}</span>
          </div>
          <div className="flex items-center space-x-2">
            {isLive && (
              <span className="flex items-center px-2 sm:px-3 py-1 rounded-full text-xs font-bold bg-red-500 text-white">
                <span className="w-2 h-2 bg-white rounded-full mr-1 sm:mr-2 animate-pulse"></span>
                LIVE
              </span>
            )}
            {isCompleted && (
              <span className="flex items-center px-2 sm:px-3 py-1 rounded-full text-xs font-bold bg-green-500 text-white">
                <CheckCircle size={12} className="mr-1 sm:w-3.5 sm:h-3.5" />
                FT
              </span>
            )}
            <span className={`px-2 sm:px-3 py-1 rounded-full text-xs font-semibold ${textSecondary}`}>
              GW {match.gameweek}
            </span>
          </div>
        </div>

        {/* Match Info: Teams and Score - Mobile Optimized */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4">
          {/* Home Team */}
          <div className="flex items-center space-x-2 sm:space-x-3 w-full sm:flex-1">
            <Image 
              src={getTeamLogo(match.home_team)} 
              alt={match.home_team}
              width={64}
              height={64}
              unoptimized
              className="rounded-lg w-10 h-10 sm:w-12 sm:h-12 md:w-16 md:h-16 flex-shrink-0 object-contain"
            />
            <div className="text-left flex-1 min-w-0">
              <p className={`font-bold ${textPrimary} text-sm sm:text-base md:text-lg truncate`}>{match.home_team}</p>
              <p className={`text-xs ${textSecondary}`}>Home</p>
              {match.home_form && <div className="hidden sm:block"><FormIndicator form={match.home_form} darkMode={darkMode} /></div>}
            </div>
          </div>

          {/* Score/VS - Centered */}
          <div className="px-2 sm:px-4 md:px-6 flex-shrink-0">
            {hasScore ? (
              <div className="text-center">
                <div className="flex items-center space-x-2 sm:space-x-3 md:space-x-4">
                  <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${textPrimary}`}>{homeScore}</span>
                  <span className={`text-lg sm:text-xl md:text-2xl ${textSecondary}`}>-</span>
                  <span className={`text-2xl sm:text-3xl md:text-4xl font-bold ${textPrimary}`}>{awayScore}</span>
                </div>
                {isCompleted && (
                  <p className={`text-xs ${textSecondary} mt-1`}>Final Score</p>
                )}
                {isLive && (
                  <p className="text-xs text-red-500 mt-1 font-bold">IN PLAY</p>
                )}
              </div>
            ) : (
              <div className="text-center">
                <span className={`text-xl sm:text-2xl font-bold ${textSecondary}`}>VS</span>
                <p className={`text-xs ${textSecondary} mt-1`}>
                  {new Date(match.date).toLocaleTimeString([], {hour: 'numeric', minute:'2-digit', hour12: true})}
                </p>
              </div>
            )}
          </div>

          {/* Away Team */}
          <div className="flex items-center space-x-2 sm:space-x-3 flex-row-reverse w-full sm:flex-1">
            <Image 
              src={getTeamLogo(match.away_team)} 
              alt={match.away_team}
              width={64}
              height={64}
              unoptimized
              className="rounded-lg w-10 h-10 sm:w-12 sm:h-12 md:w-16 md:h-16 flex-shrink-0 object-contain"
            />
            <div className="text-right flex-1 min-w-0">
              <p className={`font-bold ${textPrimary} text-sm sm:text-base md:text-lg truncate`}>{match.away_team}</p>
              <p className={`text-xs ${textSecondary}`}>Away</p>
              {match.away_form && <div className="hidden sm:flex justify-end"><FormIndicator form={match.away_form} darkMode={darkMode} /></div>}
            </div>
          </div>
        </div>

        {/* Form Indicators for Mobile - Below teams */}
        <div className="flex sm:hidden items-center justify-between mt-3 px-2">
          <div className="flex-1">
            {match.home_form && <FormIndicator form={match.home_form} darkMode={darkMode} />}
          </div>
          <div className="flex-1 flex justify-end">
            {match.away_form && <FormIndicator form={match.away_form} darkMode={darkMode} />}
          </div>
        </div>

        {/* Stadium */}
        {match.venue && (
          <div className={`mt-3 sm:mt-4 text-xs sm:text-sm ${textSecondary} text-center truncate`}>
            Stadium: {match.venue}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 lg:px-12 py-6 md:py-12 overflow-x-hidden">
      {/* Last Updated & Gameweek Navigation */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 sm:mb-8 gap-4">
        <div className={`text-xs sm:text-sm ${textSecondary} flex items-center space-x-2`}>
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span>Updated: {isMounted ? lastUpdated.toLocaleTimeString() : '--:--:--'}</span>
        </div>
        <button
          onClick={fetchMatches}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          <span>{loading ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>

      {/* Gameweek Selector */}
      <div className={`${cardBg} border rounded-2xl p-6 mb-8 shadow-xl`}>
        <div className="flex items-center justify-between">
          <button
            onClick={handlePrevGameweek}
            disabled={gameweek === null || gameweek <= 1}
            className="p-3 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft size={24} />
          </button>

          <div className="text-center">
            <p className={`text-sm ${textSecondary} mb-1`}>Gameweek</p>
            <p className={`text-4xl font-bold ${textPrimary}`}>{gameweek !== null ? gameweek : '--'}</p>
          </div>

          <button
            onClick={handleNextGameweek}
            disabled={gameweek === null || gameweek >= 38}
            className="p-3 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronRight size={24} />
          </button>
        </div>
      </div>

      {message && (
        <div className={`mb-6 text-center p-4 rounded-lg ${cardBg} border`}>
          <p className={`text-sm ${textSecondary}`}>{message}</p>
          <p className={`text-xs ${textSecondary} mt-2 italic opacity-75`}>*All times are shown in your local time</p>
        </div>
      )}

      {/* Match Filter */}
      <div className={`${cardBg} border rounded-xl p-4 mb-6 shadow-md`}>
        <p className={`text-sm ${textSecondary} mb-3 font-semibold`}>Filter Matches</p>
        
        {/* Mobile Dropdown */}
        <div className="md:hidden">
          <select
            value={resultFilter}
            onChange={(e) => setResultFilter(e.target.value)}
            className={`w-full px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              darkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-300'
            } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
          >
            <option value="all">All Matches</option>
            <option value="home_win">Home Win</option>
            <option value="draw">Draw</option>
            <option value="away_win">Away Win</option>
            <option value="upcoming">Upcoming</option>
          </select>
        </div>

        {/* Desktop Buttons */}
        <div className="hidden md:flex flex-wrap gap-2">
          <button
            onClick={() => setResultFilter('all')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              resultFilter === 'all'
                ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-blue-500/20`
            }`}
          >
            All Matches
          </button>
          <button
            onClick={() => setResultFilter('home_win')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              resultFilter === 'home_win'
                ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg'
                : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-green-500/20`
            }`}
          >
            Home Win
          </button>
          <button
            onClick={() => setResultFilter('draw')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              resultFilter === 'draw'
                ? 'bg-gradient-to-r from-yellow-500 to-amber-500 text-white shadow-lg'
                : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-yellow-500/20`
            }`}
          >
            Draw
          </button>
          <button
            onClick={() => setResultFilter('away_win')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              resultFilter === 'away_win'
                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg'
                : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-teal-500/20`
            }`}
          >
            Away Win
          </button>
          <button
            onClick={() => setResultFilter('upcoming')}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
              resultFilter === 'upcoming'
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                : `${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-gray-200 text-gray-700'} hover:bg-purple-500/20`
            }`}
          >
            Upcoming
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2" style={{borderColor: accentColor}}></div>
        </div>
      )}

      {!loading && (
        <div className="space-y-4">
          {(() => {
            const filteredMatches = matches.filter(match => {
              if (resultFilter === 'all') return true;
              
              const homeScore = match.home_score !== undefined ? match.home_score : match.home_goals;
              const awayScore = match.away_score !== undefined ? match.away_score : match.away_goals;
              const hasScore = homeScore !== null && homeScore !== undefined;
              
              // Upcoming matches (no score yet)
              if (resultFilter === 'upcoming') return !hasScore;
              
              // Completed matches only
              if (!hasScore) return false;
              
              if (resultFilter === 'home_win') return homeScore > awayScore;
              if (resultFilter === 'away_win') return awayScore > homeScore;
              if (resultFilter === 'draw') return homeScore === awayScore;
              return true;
            });
            
            return filteredMatches.length > 0 ? (
              filteredMatches.map((match, index) => renderMatch(match, index))
            ) : (
              <div className={`${cardBg} border rounded-xl p-12 text-center`}>
                <p className={`text-lg ${textSecondary}`}>No matches found for the selected filter</p>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
