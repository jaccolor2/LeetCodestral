import { useState } from 'react';
import Editor from '@monaco-editor/react';
import { ValidationResponse } from '../types/api';

interface CodeEditorProps {
  code: string;
  onChange: (value: string) => void;
  onRun: () => Promise<any>;
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

  return (
    <>
      {/* Main Editor */}
      <div className="h-full flex flex-col">
        {/* Header with language selector and buttons */}
        <div className="flex-none flex justify-between items-center p-2 bg-gray-800">
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="bg-gray-700 text-white px-2 py-1 rounded-md text-sm"
          >
            <option value="python">Python</option>
            <option value="javascript">JavaScript</option>
            <option value="java">Java</option>
            <option value="cpp">C++</option>
          </select>
          
          <div className="flex gap-2">
            <button
              onClick={onRun}
              disabled={isRunning}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1 rounded-md text-sm disabled:bg-gray-600"
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
          <div className="h-[25%] border-t border-gray-700 flex flex-col">
            <div className="flex-none flex justify-between items-center px-4 py-2 bg-gray-800 border-b border-gray-700">
              <span className="text-white text-sm font-medium">Output</span>
              {output && (
                <button
                  onClick={() => onChange('')}
                  className="text-gray-400 hover:text-white text-sm px-2 py-1 rounded"
                >
                  Clear
                </button>
              )}
            </div>
            <div className="flex-1 overflow-auto p-4 bg-[#1E1E1E] text-white font-mono text-sm whitespace-pre-wrap scrollbar-thin">
              {isRunning ? (
                <div className="text-yellow-400">Running code...</div>
              ) : (
                <div className={output.includes('Error') ? 'text-red-400' : 'text-green-400'}>
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