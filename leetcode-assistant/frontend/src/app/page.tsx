'use client';
import { useState } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { useChat } from './hooks/useChat';
import { useCodeExecution } from './hooks/useCodeExecution';
import { useProblem } from './hooks/useProblem';
import { ProblemDescription } from './components/ProblemDescription';
import { CodeEditor } from './components/CodeEditor';
import { ChatWindow } from './components/chat/ChatWindow';
import { api } from './services/api';
import TestResults from './components/TestResults';

export default function Home() {
  const { messages, loading, handleAskQuestion, setMessages } = useChat();
  const { code, setCode, runCode, isRunning, output } = useCodeExecution();
  const { currentProblem, validate, validating } = useProblem();
  const [isProblemPanelVisible, setIsProblemPanelVisible] = useState(true);
  const [testResults, setTestResults] = useState<any[]>([]);

  const handleGenerateTests = async () => {
    return api.generateTests(code, currentProblem.id);
  };

  const handleRunTests = async () => {
    try {
      const response = await api.runTests(code, currentProblem.id);
      setTestResults(response.results);
      return response;
    } catch (error) {
      console.error('Error running tests:', error);
      return { results: [] };
    }
  };

  return (
    <MainLayout
      isLeftPanelVisible={isProblemPanelVisible}
      onToggleLeftPanel={() => setIsProblemPanelVisible(!isProblemPanelVisible)}
      leftPanel={
        <ProblemDescription 
          problem={currentProblem}
          code={code}
          onRunTests={handleRunTests}
        />
      }
      centerPanel={
        <CodeEditor
          code={code}
          onChange={setCode}
          onRun={runCode}
          onValidate={validate}
          isRunning={isRunning}
          validating={validating}
          output={output}
        />
      }
      rightPanel={
        <ChatWindow
          messages={messages}
          loading={loading}
          onSend={(question) => handleAskQuestion(question, code)}
          setMessages={setMessages}
        />
      }
    >
      <div className="flex flex-col gap-4">
        <button 
          onClick={handleRunTests}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Run Tests
        </button>
        {testResults.length > 0 && <TestResults results={testResults} />}
      </div>
    </MainLayout>
  );
}
