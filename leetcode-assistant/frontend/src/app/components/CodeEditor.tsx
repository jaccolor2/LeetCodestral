import { useState } from 'react';
import Editor from '@monaco-editor/react';
import { ValidationResponse } from '../types/api';

interface CodeEditorProps {
  code: string;
  onChange: (value: string) => void;
  onRun: (language: string) => Promise<any>;
  isRunning: boolean;
  output: string;
  onValidate?: (code: string, problemId: number) => Promise<ValidationResponse>;
  validating?: boolean;
}

export function CodeEditor({ 
  code, 
  onChange, 
  onRun, 
  isRunning,
  output 
}: CodeEditorProps) {
  const [language, setLanguage] = useState('python');
  const [pythonCode, setPythonCode] = useState('');
  const [javascriptCode, setJavascriptCode] = useState('');

  const handleLanguageChange = (newLanguage: string) => {
    if (language === 'python') {
      setPythonCode(code);
    } else if (language === 'javascript') {
      setJavascriptCode(code);
    }

    setLanguage(newLanguage);

    if (newLanguage === 'python') {
      onChange(pythonCode);
    } else if (newLanguage === 'javascript') {
      onChange(javascriptCode);
    }
  };

  return (
    <>
      {/* Main Editor */}
      <div className="h-full flex flex-col">
        {/* Header with language selector and buttons */}
        <div className="flex-none flex justify-between items-center p-2 bg-[#1A1A1A]">
          <select
            value={language}
            onChange={(e) => handleLanguageChange(e.target.value)}
            className="bg-[#2D2D2D] text-white px-2 py-1 rounded-md text-sm"
          >
            <option value="python">Python</option>
            <option value="javascript">JavaScript</option>
          </select>
          
          <div className="flex gap-2">
            <button
              onClick={() => onRun(language)}
              disabled={isRunning}
              className="bg-[#FF4405] hover:bg-[#FF4405]/80 text-white px-4 py-1 rounded-md text-sm disabled:bg-gray-600"
            >
              {isRunning ? 'Running...' : 'Run Code'}
            </button>
          </div>
        </div>
        
        {/* Main content area */}
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {/* Code editor section */}
          <div className="h-[75%]">
            <Editor
              height="100%"
              language={language}
              value={code}
              onChange={(value) => onChange(value || '')}
              theme="vs-dark"
              options={{
                fontSize: 14,
                fontWeight: 'normal',
                lineNumbers: 'on',
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                automaticLayout: true,
              }}
            />
          </div>
          
          {/* Output Panel */}
          <div className="h-[25%] border-t border-[#2D2D2D] flex flex-col">
            <div className="flex-none flex justify-between items-center px-4 py-2 bg-[#1A1A1A] border-b border-[#2D2D2D]">
              <span className="text-white text-sm font-medium">Output</span>
              {output && (
                <button
                  onClick={() => onChange('')}
                  className="text-[#FF4405] hover:text-[#FF4405]/80 text-sm px-2 py-1 rounded"
                >
                  Clear
                </button>
              )}
            </div>
            <div className="flex-1 overflow-auto p-4 bg-[#1A1A1A] text-white font-mono text-sm whitespace-pre-wrap scrollbar-thin">
              {isRunning ? (
                <div className="text-[#FF4405]">Running code...</div>
              ) : (
                <div className={output.includes('Error') ? 'text-red-400' : 'text-[#FF4405]'}>
                  {output || 'Run your code to see the output here'}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
} 