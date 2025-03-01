import { useState } from 'react';
import { api } from '../services/api';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  id?: number;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);

  const handleAskQuestion = async (
    question: string,
    code: string,
    problemId: number | null,
    testResults: Array<{ passed: boolean; output: string; expected: string; error?: string }> = [],
    problem: any = null
  ) => {
    if (!problemId) return;

    const messageId = Date.now();
    setMessages(prev => [...prev, { role: 'user', content: question, id: messageId }]);
    setMessages(prev => [...prev, { role: 'assistant', content: '', id: messageId + 1 }]);

    try {
      setLoading(true);
      let fullResponse = '';

      await api.chat({
        message: question,
        code,
        problem_id: problemId,
        history: messages,
        testResults,
        problem
      }, (chunk) => {
        fullResponse += chunk;
        setMessages(prev => prev.map(msg => 
          msg.id === messageId + 1 
            ? { ...msg, content: fullResponse }
            : msg
        ));
      });
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => prev.map(msg => 
        msg.id === messageId + 1
          ? { ...msg, content: 'Sorry, there was an error. Please try again.' }
          : msg
      ));
    } finally {
      setLoading(false);
    }
  };

  return {
    messages,
    loading,
    handleAskQuestion,
    setMessages
  };
}