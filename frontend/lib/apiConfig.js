/**
 * Centralized API Configuration
 * Handles API base URL with fallback support and authentication headers
 */

// Backend URL priority list (tries in order)
const BACKEND_URLS = [
  process.env.NEXT_PUBLIC_API_URL,                     // Custom env variable (highest priority)
  'http://localhost:8000',                              // Local development
  'https://bhakundo-backend.onrender.com',              // Production Render deployment
];

// Filter out null/undefined and get first available
export const API_BASE_URL = BACKEND_URLS.find(url => url) || 'http://localhost:8000';

// Admin credentials for Basic Authentication
const ADMIN_USERNAME = 'bishaladmin';
const ADMIN_PASSWORD = 'plbishal3268';

// Track working backend URL
let workingBackendUrl = API_BASE_URL;

/**
 * Get default headers including Basic Auth for authentication
 */
export const getApiHeaders = () => {
  return {
    'Content-Type': 'application/json',
    'Authorization': 'Basic ' + btoa(`${ADMIN_USERNAME}:${ADMIN_PASSWORD}`),
  };
};

/**
 * Test if a backend URL is accessible
 */
const testBackendUrl = async (baseUrl) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout
    
    const response = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    return false;
  }
};

/**
 * Find working backend URL from the list
 */
export const findWorkingBackend = async () => {
  for (const url of BACKEND_URLS.filter(u => u)) {
    console.log(`Testing backend: ${url}`);
    const isWorking = await testBackendUrl(url);
    if (isWorking) {
      console.log(`✅ Using backend: ${url}`);
      workingBackendUrl = url;
      return url;
    }
  }
  console.warn('⚠️ No working backend found, using default');
  return workingBackendUrl;
};

/**
 * Fetch wrapper with automatic authentication headers and fallback support
 */
export const apiFetch = async (endpoint, options = {}) => {
  const tryFetch = async (baseUrl) => {
    const url = endpoint.startsWith('http') ? endpoint : `${baseUrl}${endpoint}`;
    
    const defaultOptions = {
      headers: getApiHeaders(),
    };

    // Merge options with defaults
    const mergedOptions = {
      ...defaultOptions,
      ...options,
      headers: {
        ...defaultOptions.headers,
        ...(options.headers || {}),
      },
    };

    const response = await fetch(url, mergedOptions);
    
    // Handle 401 errors specially
    if (response.status === 401) {
      console.error('API Authentication failed. Please check your API key.');
      throw new Error('Authentication failed');
    }
    
    return response;
  };

  // Try with current working backend
  try {
    return await tryFetch(workingBackendUrl);
  } catch (error) {
    console.warn(`Failed to fetch from ${workingBackendUrl}, trying fallbacks...`);
    
    // Try all backend URLs
    for (const backendUrl of BACKEND_URLS.filter(u => u && u !== workingBackendUrl)) {
      try {
        console.log(`Trying fallback: ${backendUrl}`);
        const response = await tryFetch(backendUrl);
        workingBackendUrl = backendUrl; // Update working URL
        console.log(`✅ Switched to backend: ${backendUrl}`);
        return response;
      } catch (err) {
        console.warn(`Fallback ${backendUrl} also failed`);
      }
    }
    
    // All backends failed
    console.error('All backend URLs failed');
    throw error;
  }
};
