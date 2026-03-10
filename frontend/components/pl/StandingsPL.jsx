'use client';

import React, { useState, useEffect } from 'react';
import { Award, RefreshCw } from 'lucide-react';
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

const getFormColor = (result) => {
  if (result === 'W') return 'bg-green-500';
  if (result === 'D') return 'bg-yellow-500';
  return 'bg-red-500';
};

export default function Standings({ darkMode }) {
  const [standings, setStandings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState('');
  const [isMounted, setIsMounted] = useState(false);

  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';

  // Auto-refresh every 60 seconds
  useEffect(() => {
    setIsMounted(true);
    fetchStandings();
    
    const interval = setInterval(() => {
      fetchStandings();
    }, 60000); // 60 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchStandings = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/standings`, {
        headers: getApiHeaders()
      });
      const data = await response.json();
      setStandings(data.standings);
      setLastUpdated(new Date(data.last_updated).toLocaleString());
    } catch (err) {
      console.error('Failed to load standings:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="px-4 sm:px-6 md:px-8 lg:px-12 py-6 md:py-12 overflow-x-hidden">
      <div className="max-w-7xl mx-auto">
        <div className={`${cardBg} border rounded-xl md:rounded-2xl p-4 sm:p-6 shadow-xl`}>
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3">
            <h2 className={`text-xl sm:text-2xl md:text-3xl font-bold ${textPrimary} flex items-center space-x-2 sm:space-x-3`}>
              <Award className="text-yellow-500 w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8" />
              <span className="truncate">Premier League 2025-26</span>
            </h2>
            <button 
              onClick={fetchStandings}
              disabled={loading}
              className="flex items-center space-x-2 px-3 sm:px-4 py-2 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-lg hover:shadow-lg transition-all disabled:opacity-50 text-sm flex-shrink-0"
            >
              <RefreshCw size={14} className={`sm:w-4 sm:h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>

          {isMounted && lastUpdated && (
            <div className={`text-xs sm:text-sm ${textSecondary} mb-3 sm:mb-4 flex flex-col sm:flex-row items-start sm:items-center space-y-1 sm:space-y-0 sm:space-x-2`}>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="truncate">Last updated: {lastUpdated}</span>
              </div>
              <span className="text-xs opacity-70">(Auto-refreshes every 60s)</span>
            </div>
          )}

          <div className="overflow-x-auto -mx-2 sm:mx-0">
            <div className="inline-block min-w-full align-middle">
              <table className="min-w-full">
                <thead>
                  <tr className={`border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                    <th className={`text-left py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold sticky left-0 ${cardBg} z-10`}>#</th>
                    <th className={`text-left py-2 sm:py-3 px-2 sm:px-4 ${textSecondary} text-xs sm:text-sm font-bold sticky left-6 sm:left-8 ${cardBg} z-10 min-w-[140px] sm:min-w-[180px]`}>Team</th>
                    <th className={`text-center py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold`}>P</th>
                    <th className={`text-center py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold`}>W</th>
                    <th className={`text-center py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold`}>D</th>
                    <th className={`text-center py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold`}>L</th>
                    <th className={`text-center py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold`}>GD</th>
                    <th className={`text-center py-2 sm:py-3 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm font-bold`}>Pts</th>
                    <th className={`text-left py-2 sm:py-3 px-2 sm:px-4 ${textSecondary} text-xs sm:text-sm font-bold min-w-[120px]`}>Form</th>
                  </tr>
                </thead>
                <tbody>
                  {standings.map((team, idx) => (
                    <tr 
                      key={idx}
                      className={`border-b ${darkMode ? 'border-gray-700/50' : 'border-gray-200'} hover:bg-blue-500/5 transition-colors ${
                        team.position <= 4 ? 'bg-blue-500/10' : 
                        team.position <= 5 ? 'bg-orange-500/10' : 
                        team.position >= 18 ? 'bg-red-500/10' : ''
                      }`}
                    >
                      <td className={`py-2 sm:py-4 px-1 sm:px-2 font-bold ${textPrimary} text-sm sm:text-base md:text-lg sticky left-0 ${cardBg} z-10`}>{team.position}</td>
                      <td className={`py-2 sm:py-4 px-2 sm:px-4 sticky left-6 sm:left-8 ${cardBg} z-10`}>
                        <div className="flex items-center space-x-2 sm:space-x-3">
                          <Image 
                            src={getTeamLogo(team.team)} 
                            alt={team.team}
                            width={32}
                            height={32}
                            unoptimized
                            className="rounded w-6 h-6 sm:w-8 sm:h-8 flex-shrink-0"
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                          <span className={`${textPrimary} font-medium text-xs sm:text-sm md:text-base truncate max-w-[100px] sm:max-w-none`}>{team.team}</span>
                        </div>
                      </td>
                      <td className={`text-center py-2 sm:py-4 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm`}>{team.played}</td>
                      <td className={`text-center py-2 sm:py-4 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm`}>{team.won}</td>
                      <td className={`text-center py-2 sm:py-4 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm`}>{team.drawn}</td>
                      <td className={`text-center py-2 sm:py-4 px-1 sm:px-2 ${textSecondary} text-xs sm:text-sm`}>{team.lost}</td>
                      <td className={`text-center py-2 sm:py-4 px-1 sm:px-2 ${textSecondary} font-medium text-xs sm:text-sm`}>
                        {team.goal_difference > 0 ? '+' : ''}{team.goal_difference}
                      </td>
                      <td className={`text-center py-2 sm:py-4 px-1 sm:px-2 font-bold ${textPrimary} text-sm sm:text-base md:text-lg`}>{team.points}</td>
                      <td className="py-2 sm:py-4 px-2 sm:px-4">
                        <div className="flex space-x-1">
                          {Array.isArray(team.form) && team.form.map((result, i) => (
                            <div key={i} className={`w-5 h-5 sm:w-6 sm:h-6 ${getFormColor(result)} rounded flex items-center justify-center text-white text-xs font-bold shadow-sm`}>
                              {result}
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 sm:mt-6 flex flex-col sm:flex-row items-start sm:items-center justify-center space-y-2 sm:space-y-0 sm:space-x-6 md:space-x-8 text-xs sm:text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 sm:w-4 sm:h-4 bg-blue-500 rounded flex-shrink-0"></div>
              <span className={textSecondary}>Champions League (1-4)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 sm:w-4 sm:h-4 bg-orange-500 rounded flex-shrink-0"></div>
              <span className={textSecondary}>Europa League (5)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 sm:w-4 sm:h-4 bg-red-500 rounded flex-shrink-0"></div>
              <span className={textSecondary}>Relegation (18-20)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
