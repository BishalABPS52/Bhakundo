import React, { useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import { ChevronDown, Home as HomeIcon, Trophy, BarChart3, HelpCircle, Moon, Sun, Menu, X, Calendar } from 'lucide-react';
import StandingsPL from '../components/pl/StandingsPL';
import UnderDevelopment from '../components/UnderDevelopment';

export default function Standings() {
  const [darkMode, setDarkMode] = useState(true);
  const [selectedLeague, setSelectedLeague] = useState('Premier League');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const leagues = [
    { name: 'Premier League', flag: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', available: true },
    { name: 'La Liga', flag: '🇪🇸', available: false },
    { name: 'Serie A', flag: '🇮🇹', available: false },
    { name: 'Bundesliga', flag: '🇩🇪', available: false },
    { name: 'Ligue 1', flag: '🇫🇷', available: false },
    { name: 'Champions League', flag: '🏆', available: false }
  ];

  const bgClass = darkMode ? 'bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900' : 'bg-gradient-to-br from-blue-50 via-white to-blue-50';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';
  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';

  const navItems = [
    { id: 'home', label: 'Home', icon: HomeIcon, href: '/' },
    { id: 'predictor', label: 'Predictor', icon: Trophy, href: '/predictor' },
    { id: 'standings', label: 'Standings', icon: BarChart3, href: '/standings' },
    { id: 'fixtures', label: 'Fixtures & Results', icon: Calendar, href: '/fixtures' },
    { id: 'help', label: 'Help', icon: HelpCircle, href: '/' },
  ];

  const handleLeagueSelect = (league) => {
    setSelectedLeague(league.name);
    setDropdownOpen(false);
  };

  return (
    <>
      <Head>
        <title>League Standings - Bhakundo</title>
        <meta name="description" content="Premier League standings and table" />
      </Head>

      <div className={`min-h-screen ${bgClass} transition-colors duration-300`}>
        {/* Top Navbar */}
        <nav className={`${cardBg} border-b shadow-lg sticky top-0 z-50`}>
          <div className="container mx-auto px-4 sm:px-6">
            <div className="flex items-center justify-between h-24">
              {/* Logo & Brand */}
              <Link href="/" className="flex items-center space-x-4 cursor-pointer group">
                <div className="relative">
                  <Image 
                    src="/bhakundo.png" 
                    alt="Bhakundo" 
                    width={160} 
                    height={240} 
                    className="transition-transform group-hover:scale-105 drop-shadow-lg relative z-10"
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </Link>

              {/* Desktop Navigation */}
              <div className="hidden md:flex items-center space-x-2">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = item.id === 'standings';
                  return (
                    <Link key={item.id} href={item.href}>
                      <button
                        className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                          isActive
                            ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                            : `${textSecondary} hover:bg-blue-500/10`
                        }`}
                      >
                        <Icon size={18} />
                        <span>{item.label}</span>
                      </button>
                    </Link>
                  );
                })}
              </div>

              {/* Right Side Buttons */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className="p-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:shadow-lg transition-all"
                >
                  {darkMode ? <Sun size={20} /> : <Moon size={20} />}
                </button>
                <button
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                  className="md:hidden p-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 text-white"
                >
                  {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
                </button>
              </div>
            </div>

            {/* Mobile Menu */}
            {mobileMenuOpen && (
              <div className="md:hidden py-4 space-y-2 border-t border-slate-700">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = item.id === 'standings';
                  return (
                    <Link key={item.id} href={item.href}>
                      <button
                        onClick={() => setMobileMenuOpen(false)}
                        className={`w-full flex items-center space-x-2 px-4 py-3 rounded-lg transition-all ${
                          isActive
                            ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white'
                            : `${textSecondary} hover:bg-blue-500/10`
                        }`}
                      >
                        <Icon size={18} />
                        <span>{item.label}</span>
                      </button>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        </nav>

        <div>
          {/* Header with League Selector */}
          <div className="px-6 md:px-12 pt-8 pb-4">
            <div className="max-w-7xl mx-auto">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h1 className={`text-4xl font-bold ${textPrimary} mb-2`}>League Standings</h1>
                  <p className={`text-lg ${textSecondary}`}>2025-2026 Season</p>
                </div>
              </div>

              {/* League Selector */}
              <div className="relative inline-block">
                <button
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  className={`${cardBg} border rounded-xl px-6 py-3 flex items-center space-x-3 hover:shadow-lg transition-all min-w-[250px]`}
                >
                  <span className="text-2xl">
                    {leagues.find(l => l.name === selectedLeague)?.flag}
                  </span>
                  <span className={`font-semibold ${textPrimary}`}>
                    {selectedLeague}
                  </span>
                  <ChevronDown size={20} className={`ml-auto ${textSecondary} ${dropdownOpen ? 'rotate-180' : ''} transition-transform`} />
                </button>

                {dropdownOpen && (
                  <div className={`absolute top-full left-0 mt-2 ${cardBg} border rounded-xl shadow-xl z-50 min-w-[250px] overflow-hidden`}>
                    {leagues.map((league, index) => (
                      <button
                        key={index}
                        onClick={() => handleLeagueSelect(league)}
                        className={`w-full px-6 py-3 flex items-center space-x-3 hover:bg-blue-500/10 transition-all ${
                          selectedLeague === league.name ? 'bg-blue-500/20' : ''
                        } ${!league.available ? 'opacity-60' : ''}`}
                      >
                        <span className="text-2xl">{league.flag}</span>
                        <div className="flex-1 text-left">
                          <span className={`font-semibold ${textPrimary} block`}>
                            {league.name}
                          </span>
                          {!league.available && (
                            <span className="text-xs text-yellow-500">Under Development</span>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Content */}
          {selectedLeague === 'Premier League' ? (
            <StandingsPL darkMode={darkMode} />
          ) : (
            <UnderDevelopment darkMode={darkMode} leagueName={selectedLeague} />
          )}
        </div>
      </div>
    </>
  );
}
