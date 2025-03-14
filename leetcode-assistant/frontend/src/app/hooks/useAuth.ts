import { useState, useEffect } from 'react';

export function useAuth() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('access_token');
      setIsLoggedIn(!!token);
      setIsLoading(false);
    };

    checkAuth();
    
    // Listen for storage events (in case token changes in another tab)
    window.addEventListener('storage', checkAuth);
    
    return () => {
      window.removeEventListener('storage', checkAuth);
    };
  }, []);

  // Add logout function
  const logout = () => {
    localStorage.removeItem('access_token');
    setIsLoggedIn(false);
  };

  return {
    isLoggedIn,
    setIsLoggedIn,
    isLoading,
    logout
  };
} 