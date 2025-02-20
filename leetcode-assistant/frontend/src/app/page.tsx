'use client';
import { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import { api } from './services/api';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle
} from "react-resizable-panels";
import { LoadingDots } from './components/LoadingDots';
import { Navbar } from './components/Navbar';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  id?: number;
}

interface CodeOutput {
  stdout: string;
  stderr: string;
  error?: string;
}

interface ValidationResponse {
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

interface Problem {
  id: number;
  title: string;
  difficulty: string;
  description: string;
  examples: { input: string; output: string }[];
}

export default function Home() {
  const [code, setCode] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [codeOutput, setCodeOutput] = useState<CodeOutput | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
  const [currentProblem, setCurrentProblem] = useState<Problem>({
    id: 1,
    title: "Two Sum",
    difficulty: "Easy",
    description: "Given an array of integers nums and an integer target, return indices of the two numbers...",
    examples: [
      {
        input: "nums = [2,7,11,15], target = 9",
        output: "[0,1]"
      }
    ]
  });

  // Load messages from localStorage when component mounts
  useEffect(() => {
    try {
      const savedMessages = localStorage.getItem('chatMessages');
      const savedCode = localStorage.getItem('editorCode');
      
      if (savedMessages) {
        const parsedMessages = JSON.parse(savedMessages);
        if (Array.isArray(parsedMessages)) {
          setMessages(parsedMessages);
        }
      }
      
      if (savedCode) {
        setCode(savedCode);
      }
    } catch (error) {
      console.error('Error loading from localStorage:', error);
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    try {
      if (messages.length > 0) {
        localStorage.setItem('chatMessages', JSON.stringify(messages));
      }
    } catch (error) {
      console.error('Error saving messages to localStorage:', error);
    }
  }, [messages]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Save code to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('editorCode', code);
    } catch (error) {
      console.error('Error saving code to localStorage:', error);
    }
  }, [code]);

  const handleAskQuestion = async () => {
    if (!question.trim()) return;
    
    const loadingId = Date.now();
    
    try {
      setLoading(true);
      const currentQuestion = question;
      setQuestion('');
      
      const userMessage: Message = { 
        role: 'user', 
        content: currentQuestion 
      };
      setMessages(prev => [...prev, userMessage]);

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '',
        id: loadingId 
      }]);
      
      let fullResponse = '';
      await api.chat(
        {
          message: currentQuestion,
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
      setMessages(prev => prev.filter(msg => msg.id !== loadingId));
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Error occurred while fetching response' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setMessages([]);
    localStorage.removeItem('chatMessages');
  };

  const runCode = async () => {
    setIsRunning(true);
    try {
      const result = await api.execute(code);
      setCodeOutput(result);
      
      // If there's output, automatically send it to the chat
      if (result.stdout || result.stderr || result.error) {
        const outputMessage = `Here's the output of my code:
\`\`\`
${result.stdout || ''}${result.stderr || ''}${result.error || ''}
\`\`\`
Can you help me understand what's happening?`;
        
        setQuestion(outputMessage);
        await handleAskQuestion();
      }
    } catch (error) {
      console.error('Error running code:', error);
      setCodeOutput({
        stdout: '',
        stderr: '',
        error: 'Failed to execute code'
      });
    } finally {
      setIsRunning(false);
    }
  };

  const handleValidate = async () => {
    setValidating(true);
    try {
      const result = await api.validate(code, currentProblem.id);
      setValidationResult(result);
      
      if (result.classification === 'CORRECT' && result.nextProblem) {
        // Show success message and option to move to next problem
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `🎉 Great job! You've solved this problem correctly! Would you like to try the next problem?`
        }]);
      } else {
        // Show what needs to be fixed
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Let's look at what needs improvement:\n\n${result.reason}`
        }]);
      }
    } catch (error) {
      console.error('Error validating code:', error);
    } finally {
      setValidating(false);
    }
  };

  return (
    <>
      <Navbar />
      <main className="fixed inset-0 mt-14 flex flex-col bg-gray-900">
        <div className="flex-1 p-4">
          <PanelGroup direction="vertical" className="h-full">
            {/* Top section (Problem + Editor + Output) */}
            <Panel defaultSize={30} className="overflow-hidden">
              <PanelGroup direction="horizontal" className="h-full">
                {/* Problem Description */}
                <Panel defaultSize={30} className="overflow-auto">
                  <div className="h-full border border-gray-700 rounded-lg p-4 bg-gray-800">
                    <h1 className="text-2xl font-bold mb-4 text-white">Two Sum</h1>
                    <p className="text-gray-200">
                      Given an array of integers nums and an integer target, return indices of the two numbers...
                    </p>
                  </div>
                </Panel>

                <PanelResizeHandle className="w-2 bg-gray-800 hover:bg-gray-700 transition-colors" />

                {/* Code Editor and Output */}
                <Panel defaultSize={70}>
                  <PanelGroup direction="vertical" className="h-full">
                    {/* Code Editor */}
                    <Panel defaultSize={70} className="overflow-hidden">
                      <div className="h-full flex flex-col border border-gray-700 rounded-lg">
                        <div className="flex justify-end gap-2 p-2 bg-gray-800 border-b border-gray-700">
                          <button
                            onClick={runCode}
                            disabled={isRunning}
                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1 rounded-md text-sm disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                          >
                            {isRunning ? 'Running...' : 'Run Code'}
                          </button>
                          <button
                            onClick={handleValidate}
                            disabled={validating}
                            className="bg-green-600 hover:bg-green-700 text-white px-4 py-1 rounded-md text-sm disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                          >
                            {validating ? 'Validating...' : 'Validate & Submit'}
                          </button>
                        </div>
                        <Editor
                          height="calc(100% - 40px)"
                          defaultLanguage="python"
                          value={code}
                          onChange={(value) => setCode(value || '')}
                          theme="vs-dark"
                          options={{
                            fontSize: 14,
                            fontWeight: 'normal',
                          }}
                        />
                      </div>
                    </Panel>

                    <PanelResizeHandle className="h-2 bg-gray-800 hover:bg-gray-700 transition-colors" />

                    {/* Output Panel */}
                    <Panel defaultSize={30} className="overflow-hidden">
                      <div className="h-full border border-gray-700 rounded-lg bg-gray-800 p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-white font-medium">Output</h3>
                          {codeOutput && (
                            <button
                              onClick={() => setCodeOutput(null)}
                              className="text-gray-400 hover:text-white text-sm"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                        <div className="h-[calc(100%-2rem)] overflow-auto bg-gray-900 rounded p-3 font-mono text-sm">
                          {codeOutput ? (
                            <div>
                              {codeOutput.stdout && (
                                <div className="text-green-400">
                                  <pre className="whitespace-pre-wrap">{codeOutput.stdout}</pre>
                                </div>
                              )}
                              {codeOutput.stderr && (
                                <div className="text-red-400">
                                  <pre className="whitespace-pre-wrap">{codeOutput.stderr}</pre>
                                </div>
                              )}
                              {codeOutput.error && (
                                <div className="text-yellow-400">
                                  <pre className="whitespace-pre-wrap">{codeOutput.error}</pre>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="text-gray-400 italic">No output yet. Run your code to see results.</div>
                          )}
                        </div>
                      </div>
                    </Panel>
                  </PanelGroup>
                </Panel>
              </PanelGroup>
            </Panel>

            <PanelResizeHandle className="h-2 bg-gray-800 hover:bg-gray-700 transition-colors" />

            {/* Chat Interface */}
            <Panel defaultSize={70} className="overflow-hidden">
              <div className="h-full flex flex-col border border-gray-700 rounded-lg p-4 bg-gray-800">
                <div className="flex justify-between items-center mb-2">
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
                <div 
                  ref={chatContainerRef}
                  className="flex-1 overflow-y-auto mb-4 p-4 border border-gray-700 rounded bg-gray-900 custom-scrollbar"
                >
                  {messages.map((message, index) => (
                    <div 
                      key={index} 
                      className={`p-3 rounded-2xl mb-4 ${
                        message.role === 'user' 
                          ? 'bg-[#2A2A2A] ml-auto w-fit max-w-[80%] rounded-tr-sm' 
                          : 'bg-[#1E1E1E] mr-auto w-[50%] rounded-tl-sm flex items-start gap-3'
                      } text-white`}
                    >
                      {message.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-lg bg-orange-600 flex-shrink-0 flex items-center justify-center">
                          <span className="text-sm font-medium">M</span>
                        </div>
                      )}
                      {message.role === 'assistant' ? (
                        <div className="prose prose-invert prose-pre:bg-gray-800 prose-pre:text-white max-w-none text-[15px]">
                          {message.content === '...' ? (
                            <LoadingDots />
                          ) : (
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                          )}
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap break-words text-[15px]">
                          {message.content}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    className="flex-1 border border-gray-700 rounded-lg px-4 py-2 bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                    placeholder="Ask for help..."
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !loading && question) {
                        handleAskQuestion();
                      }
                    }}
                  />
                  <button 
                    className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                    onClick={handleAskQuestion}
                    disabled={loading || !question}
                  >
                    {loading ? 'Sending...' : 'Send'}
                  </button>
                </div>
              </div>
            </Panel>
          </PanelGroup>
        </div>
      </main>
    </>
  );
}
