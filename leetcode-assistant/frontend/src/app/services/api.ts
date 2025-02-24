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
  async chat(
    params: { 
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
    },
    onChunk: (chunk: string) => void
  ) {
    // Add debug logs
    console.log('Sending chat request:', {
      message: params.message,
      code: params.code.slice(0, 100) + '...', // Truncate code for readability
      history: params.history,
      testResults: params.testResults
    });

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No reader available');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last partial line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;
          
          try {
            const chunk = JSON.parse(trimmedLine);
            if (chunk.content === '[DONE]') {
              return;
            }
            onChunk(chunk.content);
          } catch (e) {
            console.error('Error parsing chunk:', trimmedLine);
          }
        }
      }
      
      // Handle any remaining data in the buffer
      if (buffer) {
        try {
          const chunk = JSON.parse(buffer);
          if (chunk.content !== '[DONE]') {
            onChunk(chunk.content);
          }
        } catch (e) {
          console.error('Error parsing final chunk:', buffer);
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
};