import Editor from '@monaco-editor/react';

interface CodeEditorProps {
  code: string;
  onChange: (value: string) => void;
}

export function CodeEditor({ code, onChange }: CodeEditorProps) {
  return (
    <Editor
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