import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';

// Dynamically import Monaco Editor with no SSR
const MonacoEditor = dynamic(
  () => import('@monaco-editor/react'),
  { ssr: false }
);

interface CodeEditorProps {
  code: string;
  onChange: (value: string) => void;
}

export function CodeEditor({ code, onChange }: CodeEditorProps) {
  const [mounted, setMounted] = useState(false);

  // Only load editor after component is mounted on client-side
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    // Return a placeholder while the editor is loading
    return (
      <div className="w-full h-full bg-[#1e1e1e] flex items-center justify-center">
        <p className="text-gray-400">Loading code editor...</p>
      </div>
    );
  }

  return (
    <MonacoEditor
      height="100%"
      defaultLanguage="python"
      value={code}
      onChange={(value) => onChange(value || '')}
      theme="vs-dark"
      options={{
        fontSize: 14,
        fontWeight: 'normal',
      }}
    />
  );
} 