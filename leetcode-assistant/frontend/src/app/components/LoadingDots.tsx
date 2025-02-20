'use client';
import { useState, useEffect } from 'react';

export const LoadingDots = () => {
  const [dots, setDots] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-8 text-center">
      {dots || '.'}
    </div>
  );
}; 