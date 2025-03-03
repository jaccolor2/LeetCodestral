// This file is a worker proxy for Monaco Editor to avoid CORS issues
self.MonacoEnvironment = {
  baseUrl: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/'
};

// Redirect worker requests to CDN
self.importScripts('https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs/base/worker/workerMain.js'); 