'use client';
import { useState, useEffect } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { useChat } from './hooks/useChat';
import { useCodeExecution } from './hooks/useCodeExecution';
import { useProblem } from './hooks/useProblem';
import { ProblemDescription } from './components/ProblemDescription';
import { CodeEditor } from './components/CodeEditor';
import { ChatWindow } from './components/chat/ChatWindow';
import { api } from './services/api';
import TestResults from './components/TestResults';
import { ValidationResponse } from './types/api';

export default function Home() {
  const { messages, loading, handleAskQuestion, setMessages } = useChat();
  const { code, setCode, runCode, isRunning, output } = useCodeExecution();
  const { currentProblem, validate, validating } = useProblem();
  const [isProblemPanelVisible, setIsProblemPanelVisible] = useState(true);
  const [testResults, setTestResults] = useState<any[]>([]);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);

  const handleGenerateTests = async () => {
    if (!currentProblem) return;
    try {
      const response = await api.generateTests(code, currentProblem.id);
      setTestResults(response);
    } catch (error) {
      console.error('Error generating tests:', error);
    }
  };

  const handleRunTests = async () => {
    if (!currentProblem) {
      console.error('No problem selected');
      return { results: [] };
    }
    
    try {
      // Add debug log before API call
      console.log('Running tests with code:', code);
      
      const response = await api.runTests(code, currentProblem.id);
      console.log('Full API response:', response);
      setTestResults(response.results);

      // More detailed validation logging
      console.log('Validation state:', {
        validationReceived: !!response.validation,
        validationResult: response.validation,
        currentModalState: showSuccessModal,
      });

      if (response.validation) {
        setValidationResult(response.validation);
        // Add immediate check of the new validation result
        console.log('Setting validation result:', response.validation);
        
        if (response.validation.classification === 'CORRECT') {
          console.log('Solution is correct, showing modal');
          setShowSuccessModal(true);
        } else {
          console.log('Solution is not correct:', response.validation.classification);
        }
      } else {
        console.log('No validation received in response');
      }

      return response;
    } catch (error) {
      console.error('Error running tests:', error);
      return { results: [] };
    }
  };

  const handleSkip = () => {
    setShowSuccessModal(false);
    setValidationResult(null);
    // Add logic to move to next problem if needed
  };

  const handleKeepImproving = () => {
    setShowSuccessModal(false);
    setValidationResult(null);
  };

  // Add debug effect to monitor state changes
  useEffect(() => {
    console.log('State changed:', {
      showSuccessModal,
      validationResult,
      isCorrect: validationResult?.classification === 'CORRECT'
    });
  }, [showSuccessModal, validationResult]);

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
          isRunning={isRunning}
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
      {/* Debug element to always show current state */}
      <div className="fixed top-0 left-0 bg-black bg-opacity-50 p-2 text-white text-xs">
        Modal should show: {String(showSuccessModal && validationResult?.classification === 'CORRECT')}
      </div>

      {showSuccessModal && validationResult && validationResult.classification === 'CORRECT' && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-[9999] flex items-center justify-center"
             style={{pointerEvents: 'auto'}}>
          <div className="bg-gray-800 rounded-lg p-6 max-w-lg w-full mx-4 shadow-xl border-2 border-green-400">
            <div className="text-center">
              <div className="text-3xl text-green-400 font-bold mb-4">
                🎉 Solution Correct! 🎉
              </div>
              <div className="text-white text-lg mb-6">
                {validationResult.reason}
              </div>
              <div className="flex gap-4">
                <button
                  onClick={handleSkip}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-lg text-lg transition-colors"
                >
                  Skip to Next Problem
                </button>
                <button
                  onClick={handleKeepImproving}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg text-lg transition-colors"
                >
                  Keep Improving
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
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
