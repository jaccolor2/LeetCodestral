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
    <div className="h-full flex flex-col border border-gray-700 rounded-lg bg-gray-800">
      <div className="flex-none flex justify-between items-center p-4 border-b border-gray-700">
        <h2 className="text-white text-lg">Chat History</h2>
        {messages.length > 0 && (
          <button 
            onClick={clearHistory}
            className="text-gray-400 hover:text-white text-sm px-2 py-1 rounded"
          >
            Clear History
          </button>
        )}
      </div>
      <div className="flex-1 overflow-hidden relative">
        <div className="absolute inset-0 overflow-y-auto scrollbar-custom">
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
                  className={`flex ${
                    message.role === 'assistant' 
                      ? 'justify-start'
                      : 'justify-end'
                  }`}
                >
                  <div className={`${
                    message.role === 'assistant'
                      ? 'bg-[#1E1E1E] rounded-tl-sm w-[95%] flex items-start gap-3'
                      : 'bg-[#2A2A2A] rounded-tr-sm max-w-[80%] inline-block'
                  } p-4 rounded-2xl text-white`}>
                    {message.role === 'assistant' && (
                      <div className="w-8 h-8 rounded-lg bg-orange-600 flex-shrink-0 flex items-center justify-center">
                        <span className="text-sm font-medium">M</span>
                      </div>
                    )}
                    <div className={`${
                      message.role === 'assistant'
                        ? 'prose prose-invert prose-pre:bg-gray-800 prose-pre:text-white max-w-none text-base w-full'
                        : 'whitespace-pre-wrap break-words text-base'
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
      </div>
      <div className="flex-none p-4 border-t border-gray-700">
        <div className="flex gap-2">
          <input 
            type="text" 
            className="flex-1 border border-gray-700 rounded-lg px-4 py-2 bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
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
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
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