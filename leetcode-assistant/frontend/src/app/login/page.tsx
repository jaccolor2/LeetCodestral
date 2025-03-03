'use client';
import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';
import { MistralLogo } from '../components/MistralLogo';

// This is used to disable the static route indicator for this page
export const dynamic = 'auto';
export const dynamicParams = true;

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [sessionExpired, setSessionExpired] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setIsLoggedIn } = useAuth();

  // Check for session expiration message in URL
  useEffect(() => {
    const expired = searchParams.get('expired');
    if (expired === 'true') {
      setSessionExpired(true);
      setError('Your session has expired. Please log in again to continue.');
    }
  }, [searchParams]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSessionExpired(false);

    try {
      const response = await api.auth(email, password, isLogin);
      if (response.access_token) {
        localStorage.setItem('access_token', response.access_token);
        setIsLoggedIn(true);
        router.push('/');
        router.refresh();
      } else {
        setError('Authentication failed');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  return (
    <div className="min-h-screen bg-[#1A1A1A] flex items-center justify-center p-4">
      <div className="bg-[#2D2D2D] p-8 rounded-lg w-full max-w-md shadow-[0_4px_20px_rgba(0,0,0,0.4)] border border-white/10">
        {sessionExpired && (
          <div className="mb-6 p-3 bg-amber-900/50 border border-amber-500/50 rounded-md text-amber-200">
            <p className="text-sm font-medium">Your session has expired. Please log in again.</p>
          </div>
        )}
        
        <div className="text-center mb-8">
          <div className="flex flex-col items-center gap-4">
            <MistralLogo size={48} />
            <h1 className="text-xl font-bold text-white">LeetCodestral</h1>
          </div>
          <p className="text-lg text-white mt-4">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-white">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-md bg-black border border-gray-800 text-white placeholder-gray-600 focus:border-[#FF4405] focus:ring-[#FF4405] outline-none px-3 py-2"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-white">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-md bg-black border border-gray-800 text-white placeholder-gray-600 focus:border-[#FF4405] focus:ring-[#FF4405] outline-none px-3 py-2"
              required
            />
          </div>

          {error && <p className="text-red-400 text-sm">{error}</p>}
          
          <button
            type="submit"
            className="w-full bg-[#FF4405] text-white py-2 px-4 rounded hover:bg-[#FF4405]/80 transition-colors"
          >
            {isLogin ? 'Login' : 'Register'}
          </button>
        </form>
        
        <button
          onClick={() => setIsLogin(!isLogin)}
          className="mt-4 text-sm text-[#FF4405] hover:text-[#FF4405]/80 w-full text-center transition-colors"
        >
          {isLogin ? 'Need an account? Register' : 'Have an account? Login'}
        </button>
      </div>
    </div>
  );
} 