import React, { useEffect, useState } from 'react';
import Image from 'next/image';

const FORMATIONS = [
  '4-3-3','4-4-2','4-2-3-1','4-5-1','4-1-4-1','4-3-2-1','4-4-1-1',
  '4-2-2-2','4-1-2-1-2','4-3-1-2','3-5-2','3-4-3','3-4-2-1','3-5-1-1',
  '5-3-2','5-4-1','5-2-3','4-2-4-0',
];

const AUTO = 'Auto-Predict';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://bhakundo-backend.onrender.com';
const AUTH = 'Basic ' + btoa('bishaladmin:plbishal3268');

// Map DB team names → logo filenames
function teamLogo(name) {
  if (!name) return null;
  const map = {
    'AFC Bournemouth':              'AFC_Bournemouth.png',
    'Arsenal FC':                   'Arsenal_FC.png',
    'Aston Villa FC':               'Aston_Villa_FC.png',
    'Brentford FC':                 'Brentford_FC.png',
    'Brighton & Hove Albion FC':    'Brighton_and_Hove_Albion_FC.png',
    'Chelsea FC':                   'Chelsea_FC.png',
    'Crystal Palace FC':            'Crystal_Palace_FC.png',
    'Everton FC':                   'Everton_FC.png',
    'Fulham FC':                    'Fulham_FC.png',
    'Ipswich Town FC':              'Ipswich_Town_FC.png',
    'Sunderland FC':            'Leicester_City_FC.png',
    'Liverpool FC':                 'Liverpool_FC.png',
    'Manchester City FC':           'Manchester_City_FC.png',
    'Manchester United FC':         'Manchester_United_FC.png',
    'Newcastle United FC':          'Newcastle_United_FC.png',
    'Nottingham Forest FC':         'Nottingham_Forest_FC.png',
    'Southampton FC':               'Southampton_FC.png',
    'Tottenham Hotspur FC':         'Tottenham_Hotspur_FC.png',
    'West Ham United FC':           'West_Ham_United_FC.png',
    'Wolverhampton Wanderers FC':   'Wolverhampton_Wanderers_FC.png',
  };
  return map[name] ? `/team-logos/${map[name]}` : null;
}

function confidenceLabel(c) {
  if (!c && c !== 0) return { label: '—', color: 'text-slate-400' };
  const pct = Math.round(c * 100);
  if (pct >= 70) return { label: 'High Confidence',   color: 'text-green-400',  dot: 'bg-green-400'  };
  if (pct >= 50) return { label: 'Medium Confidence', color: 'text-yellow-400', dot: 'bg-yellow-400' };
  return             { label: 'Low Confidence',    color: 'text-red-400',   dot: 'bg-red-400'   };
}

function formatMatchDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

function formatKickoff(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
}

function Countdown({ iso }) {
  const [display, setDisplay] = useState('');
  useEffect(() => {
    const tick = () => {
      if (!iso) { setDisplay('—'); return; }
      const diff = new Date(iso) - Date.now();
      if (diff <= 0) { setDisplay('Live / Finished'); return; }
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setDisplay(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [iso]);
  return <span>{display}</span>;
}

function TeamBlock({ name, darkMode, formation, onFormation, showFormationPicker }) {
  const logo      = teamLogo(name);
  const ringColor = darkMode ? 'border-slate-600' : 'border-blue-200';
  const nameCls   = darkMode ? 'text-white'       : 'text-slate-900';
  const selectBg  = darkMode
    ? 'bg-slate-700 border-slate-600 text-slate-200'
    : 'bg-white border-blue-200 text-slate-700';

  return (
    <div className="flex flex-col items-center gap-2 flex-1 min-w-0">
      <div className={`relative w-16 h-16 sm:w-20 sm:h-20 rounded-full border-2 ${ringColor} bg-white/5 flex items-center justify-center overflow-hidden`}>
        {logo
          ? <Image src={logo} alt={name || ''} fill className="object-contain p-1" />
          : <span className="text-3xl">⚽</span>}
      </div>
      <p className={`text-center text-sm sm:text-base font-extrabold uppercase tracking-wide leading-tight ${nameCls}`}>
        {name || '—'}
      </p>
      {showFormationPicker && (
        <select
          value={formation}
          onChange={e => onFormation(e.target.value)}
          className={`w-full text-xs rounded-lg border px-2 py-1.5 mt-1 ${selectBg} focus:outline-none focus:ring-1 focus:ring-blue-500`}
        >
          <option value={AUTO}>Auto-Predict</option>
          {FORMATIONS.map(f => <option key={f} value={f}>{f}</option>)}
        </select>
      )}
    </div>
  );
}

// Normalize probabilities from either POST /predict (nested) or GET /game-of-gw (flat)
function getBaseProbs(src) {
  if (!src) return { home: 0, draw: 0, away: 0 };
  if (src.base_outcome_probabilities) return {
    home: src.base_outcome_probabilities.home_win || 0,
    draw: src.base_outcome_probabilities.draw     || 0,
    away: src.base_outcome_probabilities.away_win || 0,
  };
  return {
    home: src.base_home_prob  || 0,
    draw: src.base_draw_prob  || 0,
    away: src.base_away_prob  || 0,
  };
}

function getLineupProbs(src) {
  if (!src) return { home: 0, draw: 0, away: 0 };
  if (src.lineup_outcome_probabilities) return {
    home: src.lineup_outcome_probabilities.home_win || 0,
    draw: src.lineup_outcome_probabilities.draw     || 0,
    away: src.lineup_outcome_probabilities.away_win || 0,
  };
  return {
    home: src.lineup_home_prob  || 0,
    draw: src.lineup_draw_prob  || 0,
    away: src.lineup_away_prob  || 0,
  };
}

function getEnsembleProbs(src) {
  if (!src) return { home: 0, draw: 0, away: 0 };
  // POST /predict: outcome_probabilities may exist
  if (src.outcome_probabilities) return {
    home: src.outcome_probabilities.home_win || 0,
    draw: src.outcome_probabilities.draw     || 0,
    away: src.outcome_probabilities.away_win || 0,
  };
  return {
    home: src.ensemble_home_prob || 0,
    draw: src.ensemble_draw_prob || 0,
    away: src.ensemble_away_prob || 0,
  };
}

function getConfidenceValue(src) {
  if (!src) return null;
  const c = src.confidence;
  if (c == null) return null;
  if (typeof c === 'object' && c.value != null) return c.value; // POST /predict nested
  return typeof c === 'number' ? c : null; // flat float from DB
}

function verdictLabel(probs, homeTeam, awayTeam) {
  const { home, draw, away } = probs;
  if (home > draw && home > away) return { label: `${homeTeam} Win`, color: 'text-green-400' };
  if (away > draw && away > home) return { label: `${awayTeam} Win`, color: 'text-teal-400' };
  return { label: 'Draw', color: 'text-yellow-400' };
}

export default function GameOfGW({ darkMode }) {
  const [data,         setData]        = useState(null);
  const [loading,      setLoading]     = useState(true);
  const [error,        setError]       = useState(null);

  const [showFormation, setShowFormation] = useState(false);
  const [homeFormation, setHomeFormation] = useState(AUTO);
  const [awayFormation, setAwayFormation] = useState(AUTO);

  const [predicting,   setPredicting]  = useState(false);
  const [predResult,   setPredResult]  = useState(null);
  const [predError,    setPredError]   = useState(null);

  const cardBg  = darkMode ? 'bg-slate-800/90 border-slate-700'    : 'bg-white/90 border-blue-100';
  const innerBg = darkMode ? 'bg-slate-900/60 border-slate-700/70' : 'bg-blue-50/70 border-blue-100';
  const muted   = darkMode ? 'text-slate-400'                       : 'text-slate-500';
  const divider = darkMode ? 'border-slate-700'                     : 'border-blue-100';

  useEffect(() => {
    fetch(`${API_BASE_URL}/game-of-gw`, { headers: { Authorization: AUTH } })
      .then(r => r.json().then(body => ({ ok: r.ok, body })))
      .then(({ ok, body }) => {
        if (!ok) throw new Error(body?.detail || 'Failed');
        setData(body);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const handlePredict = async () => {
    if (!data) return;
    setPredicting(true);
    setPredResult(null);
    setPredError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: AUTH },
        body: JSON.stringify({
          home_team:      data.home_team,
          away_team:      data.away_team,
          home_formation: homeFormation,
          away_formation: awayFormation,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || 'Prediction failed');
      setPredResult(json);
    } catch (e) {
      setPredError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  if (loading) return (
    <div className={`max-w-2xl mx-auto rounded-2xl border p-10 text-center ${cardBg}`}>
      <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
  if (error || !data) return (
    <div className={`max-w-2xl mx-auto rounded-2xl border p-8 text-center ${cardBg}`}>
      <p className={`text-sm ${muted}`}>
        {(error || '').includes('No active') || error === '404'
          ? 'No Game of the GW set yet.'
          : `Could not load Game of the GW: ${error}`}
      </p>
    </div>
  );

  const isFinished = data.status === 'FINISHED';
  const kickoffISO = data.match_date;

  // Determine active prediction source
  const src = predResult || (data.base_home_prob != null ? data : null);

  // Normalize probs regardless of source shape
  const baseProbs   = getBaseProbs(src);
  const lineupProbs = getLineupProbs(src);
  const ensembleProbs = getEnsembleProbs(src);
  const hasProbs = baseProbs.home + baseProbs.draw + baseProbs.away > 0;

  // Score: POST uses predicted_score.home, GET uses predicted_home_goals
  const predHome = predResult
    ? (predResult.predicted_home_goals ?? predResult.predicted_score?.home)
    : data.predicted_home_goals;
  const predAway = predResult
    ? (predResult.predicted_away_goals ?? predResult.predicted_score?.away)
    : data.predicted_away_goals;

  const hasPred = predHome != null && predAway != null;

  // Confidence: POST returns nested object, GET returns flat float
  const confidenceVal = getConfidenceValue(src);
  const confidencePct = confidenceVal != null ? Math.round(confidenceVal * 100) : null;

  // Overall Bhakundo verdict from ensemble probs
  const overallVerdict = verdictLabel(ensembleProbs, data.home_team, data.away_team);

  return (
    <div className="max-w-2xl mx-auto">
      {/* Section label */}
      <div className="flex items-center gap-3 mb-4">
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
        <span className="text-xs font-bold uppercase tracking-widest text-blue-400"> Match of the Week</span>
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
      </div>

      <div className={`rounded-2xl border backdrop-blur-lg overflow-hidden shadow-lg ${cardBg}`}>

        {/* Pitch strip */}
        <div className="h-1.5 w-full" style={{background:'repeating-linear-gradient(90deg,#15803d 0,#15803d 20px,#16a34a 20px,#16a34a 40px)',opacity:0.7}} />

        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider text-sky-300 border border-sky-400/30 bg-sky-500/10">
            ★ Featured Match
          </span>
          <div className="text-right">
            <p className={`text-lg font-extrabold leading-none ${darkMode ? 'text-white' : 'text-slate-900'}`}>{formatKickoff(kickoffISO)}</p>
            <p className={`text-xs mt-0.5 ${muted}`}>Kickoff · GMT+5:45</p>
          </div>
        </div>

        {/* League row */}
        <div className={`flex items-center gap-2 px-5 py-2 border-b ${divider}`}>
          <span className="w-2 h-2 rounded-full bg-yellow-400 shadow-[0_0_6px_#facc15]" />
          <span className={`text-xs font-semibold uppercase tracking-wide ${muted}`}>🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League</span>
          {data.gameweek && <span className={`ml-auto text-xs ${muted}`}>Gameweek {data.gameweek}</span>}
        </div>

        {/* Teams + VS */}
        <div className="flex items-start gap-4 px-5 py-6">
          <TeamBlock
            name={data.home_team} darkMode={darkMode}
            formation={homeFormation} onFormation={setHomeFormation}
            showFormationPicker={showFormation}
          />
          <div className="flex flex-col items-center gap-1.5 flex-shrink-0 pt-2">
            <div className={`relative w-14 h-14 rounded-full border-2 ${darkMode ? 'border-slate-600 bg-slate-900/60' : 'border-blue-200 bg-blue-50'} flex items-center justify-center`}>
              <span className="text-sm font-black text-sky-400 tracking-widest">VS</span>
              <span className="absolute inset-0 rounded-full border border-blue-500/20 animate-spin [animation-duration:8s]" />
            </div>
            <p className={`text-xs text-center leading-snug ${muted}`}>{formatMatchDate(kickoffISO)}</p>
          </div>
          <TeamBlock
            name={data.away_team} darkMode={darkMode}
            formation={awayFormation} onFormation={setAwayFormation}
            showFormationPicker={showFormation}
          />
        </div>

        {/* Full time */}
        {isFinished && data.actual_home_goals != null && (
          <div className="px-5 pb-3">
            <div className={`rounded-xl border p-3 text-center ${innerBg}`}>
              <p className={`text-xs uppercase tracking-widest font-semibold mb-1 ${muted}`}>Full Time</p>
              <p className={`text-3xl font-black ${darkMode ? 'text-white' : 'text-slate-900'}`}>
                {data.actual_home_goals} – {data.actual_away_goals}
              </p>
            </div>
          </div>
        )}

        {/* Countdown */}
        {!isFinished && (
          <div className="flex items-center gap-2 px-5 pb-3">
            <span className={`text-xs uppercase tracking-wider ${muted}`}>⏱ Kicks off in</span>
            <span className="font-extrabold text-yellow-400 text-sm tracking-wider">
              <Countdown iso={kickoffISO} />
            </span>
          </div>
        )}

        {/* Prediction result — only shown after user clicks Predict */}
        {(predResult || predError) && (
          <div className="px-5 pb-4">
            {predError ? (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-center text-xs text-red-400">
                {predError}
              </div>
            ) : (
              <div className={`rounded-xl border p-4 space-y-4 ${innerBg}`}>

                {/* Bhakundo Predicts header */}
                <div className="p-2 sm:p-3 rounded-lg bg-teal-500/20 border border-teal-500/50 text-center">
                  <p className={`text-xs sm:text-sm font-semibold ${darkMode ? 'text-white' : 'text-slate-900'}`}>
                    Bhakundo Predicts:{' '}
                    <span className={overallVerdict.color}>{overallVerdict.label}</span>
                  </p>
                </div>

                {/* Predicted Score */}
                <div className="flex items-center justify-center">
                  <div className="text-center">
                    <p className={`text-xs mb-1 ${muted}`}>Predicted Score</p>
                    <div className={`flex items-center space-x-2 sm:space-x-3 p-2 sm:p-3 rounded-lg ${
                      predHome > predAway ? 'bg-green-500/20' :
                      predHome === predAway ? 'bg-yellow-500/20' :
                      'bg-teal-500/20'
                    }`}>
                      <span className={`text-2xl sm:text-3xl font-bold ${
                        predHome > predAway ? 'text-green-400' :
                        predHome === predAway ? 'text-yellow-400' :
                        darkMode ? 'text-white' : 'text-slate-900'
                      }`}>{predHome}</span>
                      <span className={`text-lg sm:text-xl ${muted}`}>-</span>
                      <span className={`text-2xl sm:text-3xl font-bold ${
                        predAway > predHome ? 'text-teal-400' :
                        predHome === predAway ? 'text-yellow-400' :
                        darkMode ? 'text-white' : 'text-slate-900'
                      }`}>{predAway}</span>
                    </div>
                  </div>
                </div>

                {/* Base Model + Lineup Model grid */}
                {hasProbs && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Base Model */}
                    <div className={`p-2 sm:p-3 rounded-lg ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                      <p className={`text-xs ${muted} mb-2 font-semibold`}>Base Model</p>
                      <div className="space-y-1 mb-2">
                        <div className="flex justify-between text-xs">
                          <span className={muted}>Home Win</span>
                          <span className="text-green-400 font-semibold">
                            {(baseProbs.home * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className={muted}>Draw</span>
                          <span className="text-yellow-400 font-semibold">
                            {(baseProbs.draw * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className={muted}>Away Win</span>
                          <span className="text-teal-400 font-semibold">
                            {(baseProbs.away * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className={`text-xs p-1.5 sm:p-2 rounded ${darkMode ? 'bg-slate-700' : 'bg-blue-100'} text-center`}>
                        {(() => {
                          const v = verdictLabel(baseProbs, data.home_team, data.away_team);
                          return <span className={`font-bold truncate ${v.color}`}>{v.label}</span>;
                        })()}
                      </div>
                    </div>

                    {/* Lineup Model */}
                    <div className={`p-2 sm:p-3 rounded-lg ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                      <p className={`text-xs ${muted} mb-2 font-semibold`}>Lineup Model</p>
                      <div className="space-y-1 mb-2">
                        <div className="flex justify-between text-xs">
                          <span className={muted}>Home Win</span>
                          <span className="text-green-400 font-semibold">
                            {(lineupProbs.home * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className={muted}>Draw</span>
                          <span className="text-yellow-400 font-semibold">
                            {(lineupProbs.draw * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className={muted}>Away Win</span>
                          <span className="text-teal-400 font-semibold">
                            {(lineupProbs.away * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className={`text-xs p-1.5 sm:p-2 rounded ${darkMode ? 'bg-slate-700' : 'bg-blue-100'} text-center`}>
                        {(() => {
                          const v = verdictLabel(lineupProbs, data.home_team, data.away_team);
                          return <span className={`font-bold truncate ${v.color}`}>{v.label}</span>;
                        })()}
                      </div>
                    </div>
                  </div>
                )}

                {/* Match status */}
                {!isFinished && (
                  <div className={`text-center py-2 px-3 rounded ${darkMode ? 'bg-slate-600/50' : 'bg-white'}`}>
                    <p className={`text-xs sm:text-sm ${muted}`}>⏳ Waiting for match to complete...</p>
                  </div>
                )}

              </div>
            )}
          </div>
        )}

        {/* Buttons */}
        <div className="flex gap-3 px-5 pb-5">
          <button
            onClick={handlePredict}
            disabled={predicting}
            className="flex-1 py-3 rounded-xl text-sm font-bold uppercase tracking-wider text-white bg-gradient-to-r from-blue-600 to-blue-500 hover:shadow-[0_6px_24px_rgba(37,99,255,0.5)] hover:-translate-y-0.5 transition-all disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:translate-y-0"
          >
            {predicting
              ? <span className="inline-flex items-center justify-center gap-2"><span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />Predicting…</span>
              : ' Predict the Match'}
          </button>
          <button
            onClick={() => setShowFormation(v => !v)}
            className={`flex-1 py-3 rounded-xl text-sm font-bold uppercase tracking-wider border transition-all ${
              showFormation
                ? 'bg-blue-500/20 border-blue-500/50 text-blue-300'
                : darkMode
                  ? 'text-slate-300 border-slate-600 hover:bg-slate-700 hover:text-white'
                  : 'text-slate-600 border-blue-200 hover:bg-blue-50 hover:text-slate-900'
            }`}
          >
             {showFormation ? 'Hide Formation' : 'Edit Formation'}
          </button>
        </div>

        {/* Stadium */}
        {data.venue && (
          <div className={`px-5 pb-3 text-xs sm:text-sm text-center ${muted}`}>
            Stadium: {data.venue}
          </div>
        )}

        {/* Reason - Game of the GW special feature */}
        {data.reason && (
          <div className="px-5 pb-5">
            <div className={`backdrop-blur-md rounded-lg p-3 text-center border ${darkMode ? 'bg-slate-700/40 border-slate-600/50 text-slate-100' : 'bg-blue-50/40 border-blue-200 text-slate-900'}`}>
              <p className="text-sm sm:text-base font-semibold mt-1">
                {data.reason}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}



