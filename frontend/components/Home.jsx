import React from 'react';
import { Send, BarChart3, TrendingUp, Award } from 'lucide-react';
import GameOfGW from './cards/GameOfGW';

// API Configuration
const API_BASE_URL = 'https://bhakundo-backend.onrender.com';
const ADMIN_USERNAME = 'bishaladmin';
const ADMIN_PASSWORD = 'plbishal3268';

// Helper function to get headers with Basic Auth
const getApiHeaders = () => ({
  'Content-Type': 'application/json',
  'Authorization': 'Basic ' + btoa(`${ADMIN_USERNAME}:${ADMIN_PASSWORD}`)
});

export default function Home({ darkMode }) {
  const [formData, setFormData] = React.useState({ email: '', message: '' });
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [submitStatus, setSubmitStatus] = React.useState(null);

  const bgClass = darkMode ? 'bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900' : 'bg-gradient-to-br from-blue-50 via-white to-blue-50';
  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';
  const accentColor = '#3B82F6';

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.email || !formData.message) {
      setSubmitStatus({ type: 'error', message: 'Please fill in all fields' });
      return;
    }

    setIsSubmitting(true);
    setSubmitStatus(null);

    try {
      const response = await fetch(`${API_BASE_URL}/send-contact`, {
        method: 'POST',
        headers: getApiHeaders(),
        body: JSON.stringify({
          email: formData.email,
          message: formData.message
        })
      });

      const data = await response.json();

      if (response.ok) {
        setSubmitStatus({ 
          type: 'success', 
          message: data.message || 'Message sent successfully! We\'ll get back to you soon.' 
        });
        setFormData({ email: '', message: '' });
      } else {
        setSubmitStatus({ 
          type: 'error', 
          message: data.detail || 'Failed to send message. Please try again.' 
        });
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setSubmitStatus({ 
        type: 'error', 
        message: 'Failed to send message. Please check your connection and try again.' 
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="px-4 sm:px-6 md:px-8 lg:px-12">
      <div className="space-y-12 sm:space-y-16 md:space-y-20 pb-12 sm:pb-16 md:pb-20">
        {/* Hero Section */}
        <div className="text-center pt-12 sm:pt-16 md:pt-20 space-y-4 sm:space-y-6 md:space-y-8">
          <div className="flex justify-center relative z-10">
            <div className="relative">
              <div className={`absolute inset-0 blur-3xl ${darkMode ? 'bg-cyan-500/20' : 'bg-blue-400/20'} rounded-full animate-pulse`}></div>
              <h1 className={`relative z-10 text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight ${darkMode ? 'text-white' : 'text-slate-900'}`}>
                Welcome to{' '}
                <span style={{ color: accentColor }}>Bhakundo</span>
                <span className={darkMode ? 'text-cyan-400' : 'text-blue-600'}> !</span>
              </h1>
            </div>
          </div>
          
          <p className={`text-base sm:text-lg md:text-xl ${textSecondary} max-w-2xl mx-auto leading-relaxed px-4`}>
              Bhakundo doesn't guess, it learns. Over 900 football matches analyzed, patterns decoded, trends spotted. Your predictions just got a serious upgrade.
            </p>
        </div>

        {/* Game of the GW */}
        <div className="max-w-6xl mx-auto px-4">
          <GameOfGW darkMode={darkMode} />
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6 md:gap-8 max-w-6xl mx-auto px-4">
          <div className={`${cardBg} border rounded-xl sm:rounded-2xl p-6 sm:p-8 hover:scale-105 transition-transform duration-300`}>
            <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-lg sm:rounded-xl flex items-center justify-center mb-4 sm:mb-6" style={{backgroundColor: `${accentColor}20`}}>
              <BarChart3 size={24} className="sm:w-8 sm:h-8" style={{color: accentColor}} />
            </div>
            <h3 className={`text-xl sm:text-2xl font-semibold mb-3 sm:mb-4 ${textPrimary}`}>Advanced Analytics</h3>
            <p className={`text-sm sm:text-base ${textSecondary}`}>
              Deep statistical analysis of team performance, player form, and historical data with 125+ features.
            </p>
          </div>

          <div className={`${cardBg} border rounded-xl sm:rounded-2xl p-6 sm:p-8 hover:scale-105 transition-transform duration-300`}>
            <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-lg sm:rounded-xl flex items-center justify-center mb-4 sm:mb-6" style={{backgroundColor: `${accentColor}20`}}>
              <TrendingUp size={24} className="sm:w-8 sm:h-8" style={{color: accentColor}} />
            </div>
            <h3 className={`text-xl sm:text-2xl font-semibold mb-3 sm:mb-4 ${textPrimary}`}>Formation Analysis</h3>
            <p className={`text-sm sm:text-base ${textSecondary}`}>
              Lineup-based predictions with tactical analysis including formations, midfield battle, and aggression scores.
            </p>
          </div>

          <div className={`${cardBg} border rounded-xl sm:rounded-2xl p-6 sm:p-8 hover:scale-105 transition-transform duration-300 sm:col-span-2 md:col-span-1`}>
            <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-lg sm:rounded-xl flex items-center justify-center mb-4 sm:mb-6" style={{backgroundColor: `${accentColor}20`}}>
              <Award size={24} className="sm:w-8 sm:h-8" style={{color: accentColor}} />
            </div>
            <h3 className={`text-xl sm:text-2xl font-semibold mb-3 sm:mb-4 ${textPrimary}`}>High Accuracy</h3>
            <p className={`text-sm sm:text-base ${textSecondary}`}>
              Trained on 1200+ professional matches.
            </p>
          </div>
        </div>

        {/* About Section */}
        <div className={`${cardBg} border rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-12 max-w-5xl mx-auto`}>
          <h2 className={`text-2xl sm:text-3xl md:text-4xl font-bold mb-4 sm:mb-6 ${textPrimary} text-center`}>About Bhakundo</h2>
          <div className={`text-sm sm:text-base md:text-lg ${textSecondary} space-y-3 sm:space-y-4 leading-relaxed`}>
            <p>
              Bhakundo started as a personal project - a curiosity about whether data could predict football.
            </p>
            <p>
              It grew into something real.
            </p>
            <p>
              Today, Bhakundo analyzes 125+ match factors - team form, player fitness, tactical setups, head-to-head history - trained on 1000+ real matches across seasons. It has seen the upsets, the comebacks, the last-minute drama.
            </p>
            <p>
              The result is high-accuracy predictions for your favorite team, delivered before kickoff.
            </p>
            <p>
              Not a guess. Just data doing what data does best.
            </p>
          </div>
        </div>

        {/* Contact Form */}
        <div className={`${cardBg} border rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-12 max-w-3xl mx-auto`}>
          <h2 className={`text-2xl sm:text-3xl md:text-4xl font-bold mb-6 sm:mb-8 ${textPrimary} text-center`}>Get In Touch</h2>
          <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
            {submitStatus && (
              <div 
                className={`p-4 rounded-xl border ${
                  submitStatus.type === 'success' 
                    ? darkMode 
                      ? 'bg-green-900/20 border-green-700 text-green-300' 
                      : 'bg-green-50 border-green-300 text-green-800'
                    : darkMode
                      ? 'bg-red-900/20 border-red-700 text-red-300'
                      : 'bg-red-50 border-red-300 text-red-800'
                }`}
              >
                {submitStatus.message}
              </div>
            )}
            <div>
              <label className={`block text-sm font-medium ${textSecondary} mb-2`}>Email</label>
              <input
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className={`w-full px-4 py-3 rounded-xl border ${darkMode ? 'bg-slate-700 border-slate-600 text-white' : 'bg-white border-slate-300 text-slate-900'} focus:outline-none focus:ring-2 focus:ring-blue-500`}
                placeholder="your.email@example.com"
                disabled={isSubmitting}
              />
            </div>
            <div>
              <label className={`block text-sm font-medium ${textSecondary} mb-2`}>Message</label>
              <textarea
                value={formData.message}
                onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                rows="4"
                className={`w-full px-4 py-3 rounded-xl border ${darkMode ? 'bg-slate-700 border-slate-600 text-white' : 'bg-white border-slate-300 text-slate-900'} focus:outline-none focus:ring-2 focus:ring-blue-500`}
                placeholder="Your message here..."
                disabled={isSubmitting}
              />
            </div>
            <button
              type="submit"
              disabled={isSubmitting}
              className={`w-full py-4 rounded-xl font-bold text-white flex items-center justify-center space-x-2 transition-all ${
                isSubmitting 
                  ? 'opacity-50 cursor-not-allowed' 
                  : 'hover:scale-105'
              }`}
              style={{backgroundColor: accentColor}}
            >
              {isSubmitting ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Sending...</span>
                </>
              ) : (
                <>
                  <Send size={20} />
                  <span>Send Message</span>
                </>
              )}
            </button>
          </form>
        </div>

      </div>
    </div>
  );
}
