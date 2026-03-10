'use client';

import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { Home, Trophy, BarChart3, HelpCircle, Moon, Sun } from 'lucide-react';

export default function Navbar({ darkMode, setDarkMode }) {
  const router = useRouter();
  const accentColor = '#3B82F6';
  
  const navItems = [
    { href: '/', icon: Home, label: 'Home' },
    { href: '/predictor', icon: Trophy, label: 'Predictor' },
    { href: '/standings', icon: BarChart3, label: 'Standings' },
    { href: '/help', icon: HelpCircle, label: 'Help' },
  ];

  const isActive = (path) => router.pathname === path;

  return (
    <nav className={`${darkMode ? 'bg-slate-800/95' : 'bg-white/95'} backdrop-blur-lg border-b ${darkMode ? 'border-slate-700' : 'border-blue-100'} sticky top-0 z-50 shadow-lg`}>
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link href="/">
            <div className="flex items-center gap-3 cursor-pointer group">
              <img 
                src="/bhakundo.png" 
                alt="Bhakundo" 
                className="w-12 h-12 transition-transform group-hover:scale-110"
              />
              <div>
                <h1 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-slate-900'}`}>
                  Bhakundo
                </h1>
                <p className="text-xs" style={{ color: accentColor }}>
                  Prepare, Predict & Play
                </p>
              </div>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);
              
              return (
                <Link key={item.href} href={item.href}>
                  <button
                    className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all ${
                      active
                        ? 'text-white shadow-lg scale-105'
                        : darkMode
                        ? 'text-slate-300 hover:bg-slate-700/50'
                        : 'text-slate-600 hover:bg-blue-50'
                    }`}
                    style={active ? { backgroundColor: accentColor } : {}}
                  >
                    <Icon size={18} />
                    <span>{item.label}</span>
                  </button>
                </Link>
              );
            })}
          </div>

          {/* Theme Toggle */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-3 rounded-xl transition-all hover:scale-110 hidden md:block"
            style={{ backgroundColor: accentColor }}
          >
            {darkMode ? (
              <Sun size={20} color="white" />
            ) : (
              <Moon size={20} color="white" />
            )}
          </button>

          {/* Mobile Menu */}
          <div className="md:hidden flex items-center gap-2">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg"
              style={{ backgroundColor: accentColor }}
            >
              {darkMode ? <Sun size={18} color="white" /> : <Moon size={18} color="white" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden pb-4">
          <div className="flex gap-2 overflow-x-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);
              
              return (
                <Link key={item.href} href={item.href}>
                  <button
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-all ${
                      active
                        ? 'text-white shadow-lg'
                        : darkMode
                        ? 'text-slate-300 bg-slate-700/50'
                        : 'text-slate-600 bg-blue-50'
                    }`}
                    style={active ? { backgroundColor: accentColor } : {}}
                  >
                    <Icon size={16} />
                    <span className="text-sm">{item.label}</span>
                  </button>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
