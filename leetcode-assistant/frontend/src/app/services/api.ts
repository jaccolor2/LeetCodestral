import { useAuth } from '../hooks/useAuth';

const API_BASE_URL = 'http://localhost:8000';

export interface ChatRequest {
  message: string;
  code: string;
  problem_id: number;
  history: Array<{ role: string; content: string }>;
  testResults?: Array<{
    passed: boolean;
    output: string;
    expected: string;
    error?: string;
  }>;
  problem?: any;
}

export interface ChatResponse {
  response: string;
}

export interface AuthResponse {
  access_token: string;
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
  async auth(email: string, password: string, isLogin: boolean): Promise<AuthResponse> {
    if (isLogin) {
      return this.login(email, password);
    } else {
      return this.register(email, password);
    }
  },

  async chat(request: ChatRequest, onMessage?: (message: string) => void): Promise<void> {
    const token = localStorage.getItem('access_token');
    if (!token) {
      throw new Error('No authentication token found');
    }

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        const lines = text.split('\n').filter(Boolean);

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.content === '[DONE]') break;
            onMessage?.(data.content);
          } catch (e) {
            console.error('Error parsing chat response:', e);
          }
        }
      }
    } finally {
      reader.releaseLock();
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

  execute: async (code: string, language: string) => {
    const response = await fetch('http://localhost:8000/api/execute', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code, language }),
    });

    if (!response.ok) {
      throw new Error('Failed to execute code');
    }

    return response.json();
  },

  validate: async (code: string, problemId: number): Promise<ValidationResponse> => {
    try {
      console.log('Sending validation request:', { code, problemId }); // Debug log

      const response = await fetch(`${API_BASE_URL}/api/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code, problem_id: problemId }),
      });

      const contentType = response.headers.get('content-type');
      console.log('Response content type:', contentType); // Debug log

      if (!response.ok) {
        let errorDetail;
        try {
          // Try to get error details from response
          errorDetail = await response.text();
          console.error('Error response body:', errorDetail);
        } catch (e) {
          errorDetail = 'No error details available';
        }

        throw new Error(`Validation failed (${response.status}): ${errorDetail}`);
      }

      const data = await response.json();
      console.log('Validation response:', data); // Debug log
      return data;
    } catch (error: any) {
      console.error('Validation error details:', {
        name: error.name,
        message: error.message,
        stack: error.stack
      });
      throw error;
    }
  },

  generateTests: async (code: string, problemId: number) => 
    fetch(`${API_BASE_URL}/api/generate-tests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, problem_id: problemId })
    }).then(res => res.json()),
    
  runTests: async (code: string, problemId: number) => {
    const response = await fetch(`${API_BASE_URL}/api/run-tests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, problem_id: problemId })
    });
    const data = await response.json();
    console.log('API Response:', data);  // Debug log
    return data;
  },

  async register(email: string, password: string) {
    const response = await fetch(`${API_BASE_URL}/api/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      throw new Error('Registration failed');
    }

    const data = await response.json();
    localStorage.setItem('access_token', data.access_token);
    return data;
  },

  async login(email: string, password: string) {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    const response = await fetch(`${API_BASE_URL}/api/token`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Login failed');
    }
    
    const data = await response.json();
    localStorage.setItem('access_token', data.access_token);
    return data;
  },
};