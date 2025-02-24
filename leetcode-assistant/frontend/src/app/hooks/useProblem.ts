import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Problem } from '../types/problem';

export function useProblem() {
  const [currentProblem, setCurrentProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    const fetchProblem = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/api/problems');
        const data = await response.json();
        
        if (data.problems && data.problems.length > 0) {
          setCurrentProblem(data.problems[0]);
          setError(null);
        } else {
          setError('No problems found');
        }
      } catch (err) {
        setError('Failed to fetch problem');
        console.error('Error fetching problem:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchProblem();
  }, []);

  const validate = async (code: string, problemId: number) => {
    setValidating(true);
    try {
      return await api.validate(code, problemId);
    } finally {
      setValidating(false);
    }
  };

  return { 
    currentProblem, 
    validate, 
    validating,
    loading,
    error 
  };
} 