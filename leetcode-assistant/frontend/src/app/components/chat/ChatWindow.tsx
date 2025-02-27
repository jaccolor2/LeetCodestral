'use client';
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { LoadingDots } from '../LoadingDots';
import { Message } from '../../types/chat';
import { format } from 'date-fns';

interface ChatWindowProps {
  messages: Message[];
  loading: boolean;
  onSend: (question: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export function ChatWindow({ messages, loading, onSend, setMessages }: ChatWindowProps) {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSend = () => {
    if (inputValue.trim()) {
      onSend(inputValue);
      setInputValue('');
    }
  };

  const clearHistory = () => {
    setMessages([]);
    localStorage.removeItem('chatMessages');
  };

  // Smooth scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Scroll on messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="h-full flex flex-col bg-[#1A1A1A]">
      <div className="flex justify-between items-center p-4 border-b border-[#2D2D2D]">
        <h2 className="text-white">Chat History</h2>
        {messages.length > 0 && (
          <button 
            onClick={clearHistory}
            className="text-[#FF4405] hover:text-[#FF4405]/80"
          >
            Clear History
          </button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-4 bg-[#1A1A1A]">
        <div className="p-4 space-y-6">
          {messages.map((message, index) => {
            // Skip duplicate messages
            if (index > 0 && 
                message.role === 'assistant' && 
                messages[index - 1].role === 'assistant' && 
                message.content === messages[index - 1].content) {
              return null;
            }

            return (
              <div 
                key={index} 
                className={`mb-4 ${
                  message.role === 'assistant' 
                    ? 'flex justify-start'
                    : 'flex justify-end'
                }`}
              >
                <div className={`${
                  message.role === 'assistant'
                    ? 'bg-[#1A1A1A] text-white w-full'
                    : 'bg-[#2D2D2D] text-white max-w-[60%]'
                } p-4 rounded-lg`}>
                  {message.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-lg bg-orange-600 flex-shrink-0 flex items-center justify-center text-white">
                      <span className="text-sm font-medium">M</span>
                    </div>
                  )}
                  <div className={`${
                    message.role === 'assistant'
                      ? 'prose prose-invert prose-pre:bg-gray-800 prose-pre:text-white max-w-none text-base w-full'
                      : 'whitespace-pre-wrap break-words text-base text-white'
                  }`}>
                    {message.role === 'assistant' ? (
                      message.content === '' ? (
                        <LoadingDots />
                      ) : (
                        <ReactMarkdown>{message.content || ''}</ReactMarkdown>
                      )
                    ) : (
                      message.content
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} /> {/* Scroll anchor */}
        </div>
      </div>
      <div className="p-4 border-t border-[#2D2D2D] bg-[#1A1A1A]">
        <div className="flex gap-2">
          <input 
            type="text" 
            className="flex-1 px-4 py-2 rounded bg-[#2D2D2D] text-white placeholder-gray-400 border border-[#2D2D2D] focus:border-[#FF4405] outline-none"
            placeholder="Ask for help..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !loading && inputValue) {
                handleSend();
              }
            }}
          />
          <button 
            className="bg-[#FF4405] text-white px-4 py-2 rounded-lg disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
            onClick={handleSend}
            disabled={loading || !inputValue}
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
} 