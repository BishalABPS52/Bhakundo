import React, { useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import UnderDevelopment from '../components/UnderDevelopment';

export default function UnderDevelopmentPage() {
  const [darkMode, setDarkMode] = useState(true);
  const router = useRouter();
  const { league } = router.query;

  const bgClass = darkMode
    ? 'bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900'
    : 'bg-gradient-to-br from-blue-50 via-white to-blue-50';

  return (
    <>
      <Head>
        <title>Coming Soon — Bhakundo</title>
        <meta name="description" content="This feature is under development." />
      </Head>
      <div className={`min-h-screen ${bgClass} transition-colors duration-300`}>
        <UnderDevelopment darkMode={darkMode} leagueName={league || 'This League'} />
      </div>
    </>
  );
}
