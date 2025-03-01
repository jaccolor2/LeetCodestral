import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../services/api';
import { Problem } from '../types/problem';

export function useProblem() {
  const [currentProblem, setCurrentProblem] = useState<Problem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  const initialLoadDone = useRef(false);

  const fetchNewProblem = useCallback(async (force: boolean = false) => {
    try {
      setLoading(true);
      const response = await api.getProblems();
      
      if (response.problems && response.problems.length > 0) {
        setCurrentProblem(response.problems[0]);
        setError(null);
        return response.problems[0];
      } else {
        setError('No problems found');
        return null;
      }
    } catch (err) {
      setError('Failed to fetch problem');
      console.error('Error fetching problem:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Only fetch on initial mount
  useEffect(() => {
    if (!initialLoadDone.current) {
      initialLoadDone.current = true;
      fetchNewProblem();
    }
  }, [fetchNewProblem]);

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
    setCurrentProblem,
    fetchNewProblem,
    validate, 
    validating,
    loading,
    error 
  };
} 