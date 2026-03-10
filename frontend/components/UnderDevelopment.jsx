import React from 'react';
import { Construction, AlertCircle } from 'lucide-react';

export default function UnderDevelopment({ darkMode, leagueName }) {
  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';

  return (
    <div className="px-6 md:px-12 py-12">
      <div className="max-w-4xl mx-auto">
        <div className={`${cardBg} border rounded-2xl p-12 text-center`}>
          <div className="flex justify-center mb-6">
            <div className="w-24 h-24 rounded-full bg-gradient-to-r from-yellow-500 to-orange-500 flex items-center justify-center shadow-xl">
              <Construction size={48} className="text-white" />
            </div>
          </div>
          
          <h1 className={`text-4xl font-bold ${textPrimary} mb-4`}>
            {leagueName} - Under Development
          </h1>
          
          <div className="flex items-center justify-center space-x-2 mb-6">
            <AlertCircle size={20} className="text-yellow-500" />
            <p className={`text-xl ${textSecondary}`}>
              This feature is currently under development
            </p>
          </div>
          
          <div className={`${darkMode ? 'bg-slate-700/50' : 'bg-blue-50'} rounded-xl p-6 mb-6`}>
            <p className={`${textSecondary} leading-relaxed mb-4`}>
              We're working hard to bring you predictions for {leagueName}!
            </p>
            <p className={`${textSecondary} text-sm`}>
              Currently, our AI models are trained on Premier League data. We're expanding our coverage to include more leagues soon.
            </p>
          </div>
          
          <div className={`${textSecondary} text-sm`}>
            <p className="mb-2">✨ Coming Soon:</p>
            <ul className="space-y-1">
              <li>• Match predictions for {leagueName}</li>
              <li>• Team statistics and analysis</li>
              <li>• League standings</li>
              <li>• Fixtures and results</li>
            </ul>
          </div>
          
          <div className="mt-8">
            <p className={`${textSecondary} text-xs`}>
              Stay tuned for updates! For now, enjoy our Premier League predictions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
