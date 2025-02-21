import { useState } from 'react';
import { api } from '../services/api';
import { Problem } from '../types/problem';

export function useProblem() {
  const [currentProblem, setCurrentProblem] = useState<Problem>({
    id: 1,
    title: "Two Sum",
    difficulty: "Easy",
    description: "Given an array of integers nums and an integer target, return indices of the two numbers...",
    examples: [
      {
        input: "nums = [2,7,11,15], target = 9",
        output: "[0,1]"
      }
    ]
  });
  const [validating, setValidating] = useState(false);

  const validate = async (code: string, problemId: number) => {
    setValidating(true);
    try {
      return await api.validate(code, problemId);
    } finally {
      setValidating(false);
    }
  };

  return { currentProblem, validate, validating };
} 