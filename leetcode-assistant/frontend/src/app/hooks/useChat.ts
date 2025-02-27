import { useState, useEffect } from 'react';
import { Message } from '../types/chat';
import { api } from '../services/api';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  // Load messages from localStorage on mount
  useEffect(() => {
    const savedMessages = localStorage.getItem('chatMessages');
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages));
  }, [messages]);

  const handleAskQuestion = async (
    question: string, 
    code: string,
    problemId: number | null,
    testResults?: Array<{ passed: boolean; output: string; expected: string; error?: string }>
  ) => {
    if (!question.trim()) return;
    
    const loadingId = Date.now();
    setMessages(prev => [...prev, 
      { role: 'user', content: question }
    ]);

    if (!problemId) {
      setMessages(prev => [...prev,
        { 
          role: 'assistant', 
          content: 'Please wait while the problem loads...',
          id: loadingId
        }
      ]);
      return;
    }
    
    try {
      setLoading(true);
      setMessages(prev => [...prev.filter(msg => msg.id !== loadingId),
        { role: 'assistant', content: '', id: loadingId }
      ]);
      
      let fullResponse = '';
      await api.chat(
        {
          message: question,
          code: code,
          problem_id: problemId,
          history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          })),
          testResults
        },
        (message) => {
          if (message === '[DONE]') return;
          fullResponse += message;
          setMessages(prev => prev.map(msg => 
            msg.id === loadingId 
              ? { ...msg, content: fullResponse }
              : msg
          ));
        }
      );
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingId),
        { 
          role: 'assistant', 
          content: 'Sorry, there was an error. Please try again in a moment.',
          timestamp: Date.now()
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return { messages, loading, handleAskQuestion, setMessages };
}