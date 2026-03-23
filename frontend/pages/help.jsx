import React, { useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import { Home as HomeIcon, Trophy, BarChart3, HelpCircle, Moon, Sun, Menu, X, Calendar } from 'lucide-react';
import Help from '../components/Help';
import Footer from '../components/Footer';

export default function HelpPage() {
  const [darkMode, setDarkMode] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const bgClass = darkMode ? 'bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900' : 'bg-gradient-to-br from-blue-50 via-white to-blue-50';
  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';

  const navItems = [
    { id: 'home', label: 'Home', icon: HomeIcon, href: '/' },
    { id: 'predictor', label: 'Predictor', icon: Trophy, href: '/predictor' },
    { id: 'standings', label: 'Standings', icon: BarChart3, href: '/standings' },
    { id: 'fixtures', label: 'Fixtures & Results', icon: Calendar, href: '/fixtures' },
    { id: 'help', label: 'Help', icon: HelpCircle, href: '/help' },
  ];

  return (
    <>
      <Head>
        <title>Help & Guide - Bhakundo</title>
        <meta name="description" content="Help center and FAQ for Bhakundo predictor" />
      </Head>

      <div className={`min-h-screen ${bgClass} transition-colors duration-300`}>
        {/* Top Navbar */}
        <nav className={`${cardBg} border-b shadow-lg sticky top-0 z-50`}>
          <div className="container mx-auto px-4 sm:px-6">
            <div className="flex items-center justify-between h-24">
              {/* Logo & Brand */}
              <Link href="/">
                <div className="flex items-center space-x-4 cursor-pointer group">
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
                </div>
              </Link>

              {/* Desktop Navigation */}
              <div className="hidden md:flex items-center space-x-2">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = item.id === 'help';
                  
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
                  const isActive = item.id === 'help';
                  
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
        
        {/* Main Content */}
        <Help darkMode={darkMode} />
        <Footer darkMode={darkMode} />
      </div>
    </>
  );
}
