// This declaration file helps prevent Monaco Editor from loading during SSR
declare module '@monaco-editor/react' {
  import React from 'react';
  
  interface EditorProps {
    height?: string | number;
    width?: string | number;
    value?: string;
    defaultValue?: string;
    language?: string;
    defaultLanguage?: string;
    theme?: string;
    options?: any;
    onChange?: (value: string | undefined) => void;
    onMount?: (editor: any, monaco: any) => void;
    beforeMount?: (monaco: any) => void;
    loading?: React.ReactNode;
    className?: string;
    wrapperClassName?: string;
  }
  
  function Editor(props: EditorProps): JSX.Element;
  
  export default Editor;
} 