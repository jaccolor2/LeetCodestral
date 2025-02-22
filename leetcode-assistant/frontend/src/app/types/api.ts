export interface ValidationResponse {
  classification: 'CORRECT' | 'INCORRECT';
  reason: string;
  testResults?: Array<{
    passed: boolean;
    output: string;
    expected: string;
    error?: string;
  }>;
  nextProblem?: number;
} 