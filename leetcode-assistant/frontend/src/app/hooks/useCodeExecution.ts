import { useState, useEffect } from 'react';
import { api } from '../services/api';

interface ExecutionResult {
  stdout: string;
  stderr: string;
  error?: string;
}

export function useCodeExecution() {
  const [code, setCode] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState<string>('');

  // Load code from localStorage on mount
  useEffect(() => {
    const savedCode = localStorage.getItem('editorCode');
    if (savedCode) {
      setCode(savedCode);
    }
  }, []);

  // Save code to localStorage whenever it changes
  useEffect(() => {
    if (code) {
      localStorage.setItem('editorCode', code);
    }
  }, [code]);

  const runCode = async (language: string) => {
    setIsRunning(true);
    setOutput('Running code...');
    try {
      const result: ExecutionResult = await api.execute(code, language);
      
      // Format the output combining stdout, stderr and error if they exist
      const formattedOutput = [
        result.stdout && `Output:\n${result.stdout}`,
        result.stderr && `Errors:\n${result.stderr}`,
        result.error && `Execution Error:\n${result.error}`
      ].filter(Boolean).join('\n\n');
      
      setOutput(formattedOutput || 'No output');
      return result;
    } catch (error) {
      setOutput(`Error: ${error instanceof Error ? error.message : 'An error occurred'}`);
    } finally {
      setIsRunning(false);
    }
  };

  const clearCode = () => {
    setCode('');
    setOutput('');
    localStorage.removeItem('editorCode');
  };

  return { code, setCode, runCode, isRunning, clearCode, output };
} 