'use client';
import { useState } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { useChat } from './hooks/useChat';
import { useCodeExecution } from './hooks/useCodeExecution';
import { useProblem } from './hooks/useProblem';
import { ProblemDescription } from './components/ProblemDescription';
import { CodeEditor } from './components/CodeEditor';
import { ChatWindow } from './components/chat/ChatWindow';

export default function Home() {
  const { messages, loading, handleAskQuestion, setMessages } = useChat();
  const { code, setCode, runCode, isRunning, output } = useCodeExecution();
  const { currentProblem, validate, validating } = useProblem();
  const [isProblemPanelVisible, setIsProblemPanelVisible] = useState(true);

  return (
    <MainLayout
      isLeftPanelVisible={isProblemPanelVisible}
      onToggleLeftPanel={() => setIsProblemPanelVisible(!isProblemPanelVisible)}
      leftPanel={<ProblemDescription problem={currentProblem} />}
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
    />
  );
}
