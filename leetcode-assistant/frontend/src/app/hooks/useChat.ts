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

  const handleAskQuestion = async (question: string, code: string) => {
    if (!question.trim()) return;
    
    const loadingId = Date.now();
    
    try {
      setLoading(true);
      
      // Combine both messages in a single update
      setMessages(prev => [...prev, 
        { role: 'user', content: question },
        { role: 'assistant', content: '', id: loadingId }
      ]);
      
      let fullResponse = '';
      await api.chat(
        {
          message: question,
          code: code,
          problem_id: 1
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
      // Remove the loading message and show error
      setMessages(prev => [
        ...prev.filter(msg => msg.id !== loadingId),
        { 
          role: 'assistant', 
          content: error instanceof Error ? error.message : 'Error occurred while fetching response',
          timestamp: Date.now()
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return { messages, loading, handleAskQuestion, setMessages };
}