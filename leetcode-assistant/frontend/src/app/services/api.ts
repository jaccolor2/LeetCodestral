const API_BASE_URL = 'http://localhost:8000';

export interface ChatRequest {
  message: string;
  code: string;
  problem_id: number;
}

export interface ChatResponse {
  response: string;
}

export interface Problem {
  id: number;
  title: string;
  difficulty: string;
  description: string;
  examples: {
    input: string;
    output: string;
  }[];
}

export interface ValidationResponse {
  classification: 'CORRECT' | 'INCORRECT';
  reason: string;
  testResults?: {
    passed: boolean;
    output: string;
    expected: string;
    error?: string;
  }[];
  nextProblem?: number;
}

export const api = {
  chat: async (data: ChatRequest, onMessage: (message: string) => void) => {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get reader from response body');
    }

    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const message = decoder.decode(value);
      onMessage(message);
    }
  },

  async getProblems(): Promise<{ problems: Problem[] }> {
    const response = await fetch(`${API_BASE_URL}/api/problems`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch problems');
    }

    return response.json();
  },

  async getProblem(id: number): Promise<Problem> {
    const response = await fetch(`${API_BASE_URL}/api/problems/${id}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch problem');
    }

    return response.json();
  },

  execute: async (code: string) => {
    const response = await fetch('http://localhost:8000/api/execute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });

    if (!response.ok) {
      throw new Error('Failed to execute code');
    }

    return response.json();
  },

  validate: async (code: string, problemId: number): Promise<ValidationResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code, problem_id: problemId }),
    });

    if (!response.ok) {
      throw new Error('Failed to validate code');
    }

    return response.json();
  }
};