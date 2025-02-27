'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';
import { MistralLogo } from '../components/MistralLogo';

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const router = useRouter();
  const { setIsLoggedIn } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      const response = await api.auth(email, password, isLogin);
      localStorage.setItem('token', response.token);
      setIsLoggedIn(true);
      router.push('/');
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  return (
    <div className="min-h-screen bg-[#1A1A1A] flex items-center justify-center p-4">
      <div className="bg-[#2D2D2D] p-8 rounded-lg w-full max-w-md">
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
              className="mt-1 block w-full rounded-md bg-[#1A1A1A] border border-[#2D2D2D] text-white focus:border-[#FF4405] focus:ring-[#FF4405] outline-none"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-white">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-md bg-[#1A1A1A] border border-[#2D2D2D] text-white focus:border-[#FF4405] focus:ring-[#FF4405] outline-none"
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