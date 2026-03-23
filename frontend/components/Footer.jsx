import Link from 'next/link';

const SOCIAL = {
  instagram: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.406-11.845a1.44 1.44 0 1 0 0 2.881 1.44 1.44 0 0 0 0-2.881z" />
    </svg>
  ),
  x: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.737-8.835L1.254 2.25H8.08l4.261 5.632 5.903-5.632Zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  ),
  facebook: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
    </svg>
  ),
};

const LEAGUES = [
  { name: 'Premier League',        href: '/standings' },
  { name: 'La Liga',               href: '/under-development?league=La+Liga' },
  { name: 'Ligue 1',               href: '/under-development?league=Ligue+1' },
  { name: 'UEFA Champions League', href: '/under-development?league=UEFA+Champions+League' },
  { name: 'Serie A',               href: '/under-development?league=Serie+A' },
  { name: 'Bundesliga',            href: '/under-development?league=Bundesliga' },
];

export default function Footer({ darkMode }) {
  const bg        = darkMode ? 'bg-slate-900/95 border-slate-700/60' : 'bg-white/95 border-blue-100';
  const heading   = darkMode ? 'text-white'       : 'text-slate-900';
  const link      = darkMode ? 'text-slate-400 hover:text-white'  : 'text-slate-500 hover:text-slate-900';
  const divider   = darkMode ? 'border-slate-700' : 'border-blue-100';
  const bottomTxt = darkMode ? 'text-slate-500'   : 'text-slate-400';
  const iconHover = darkMode ? 'hover:text-white text-slate-400' : 'hover:text-blue-600 text-slate-500';

  return (
    <footer className={`mt-20 border-t backdrop-blur-lg ${bg}`}>
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-14">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-10">

          {/* Explore Bhakundo */}
          <div>
            <h3 className={`text-sm font-semibold uppercase tracking-widest mb-5 ${heading}`}>
              Explore Bhakundo
            </h3>
            <ul className="space-y-3">
              {[
                { label: 'Home',              href: '/' },
                { label: 'Predictor',         href: '/predictor' },
                { label: 'Standings',         href: '/standings' },
                { label: 'Fixtures & Results',href: '/fixtures' },
                { label: 'Help',              href: '/help' },
              ].map(({ label, href }) => (
                <li key={label}>
                  <Link href={href}>
                    <span className={`text-sm transition-colors cursor-pointer ${link}`}>{label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Leagues */}
          <div>
            <h3 className={`text-sm font-semibold uppercase tracking-widest mb-5 ${heading}`}>
              Leagues
            </h3>
            <ul className="space-y-3">
              {LEAGUES.map(({ name, href }) => (
                <li key={name}>
                  <Link href={href}>
                    <span className={`text-sm transition-colors cursor-pointer ${link}`}>{name}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className={`text-sm font-semibold uppercase tracking-widest mb-5 ${heading}`}>
              Resources
            </h3>
            <ul className="space-y-3">
              {[
                { label: 'About',       href: '/' },
                { label: 'Help Center', href: '/help' },
                { label: 'Contact Us',  href: '/' },
                { label: "FAQ's",       href: '/help' },
              ].map(({ label, href }) => (
                <li key={label}>
                  <Link href={href}>
                    <span className={`text-sm transition-colors cursor-pointer ${link}`}>{label}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Follow Us */}
          <div>
            <h3 className={`text-sm font-semibold uppercase tracking-widest mb-5 ${heading}`}>
              Follow Us
            </h3>
            <div className="flex gap-4 mt-1">
              {[
                { label: 'Instagram', href: 'https://www.instagram.com/bs52.py/',       icon: SOCIAL.instagram },
                { label: 'X',         href: 'https://x.com/BishalS85851861',            icon: SOCIAL.x         },
                { label: 'Facebook',  href: 'https://www.facebook.com/bs52.py',         icon: SOCIAL.facebook  },
              ].map(({ label, href, icon }) => (
                <a
                  key={label}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={label}
                  className={`transition-colors ${iconHover}`}
                >
                  {icon}
                </a>
              ))}
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className={`mt-12 pt-6 border-t ${divider} flex flex-col sm:flex-row items-center justify-between gap-4`}>
          <p className={`text-xs ${bottomTxt}`}>
            Terms of Service
          </p>
          <p className={`text-xs ${bottomTxt}`}>
            Built by{' '}
            <a
              href="https://www.bishalshrestha52.com.np/"
              target="_blank"
              rel="noopener noreferrer"
              className="bulgatti text-base text-blue-500 hover:text-blue-400 transition-colors"
            >
              Bishal Shrestha
            </a>
          </p>
          <p className={`text-xs ${bottomTxt}`}>
            Bhakundo &mdash; Prepare, Predict &amp; Play
          </p>
        </div>
      </div>
    </footer>
  );
}
