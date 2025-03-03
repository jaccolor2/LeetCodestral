import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        {/* Preconnect to CDN domains for performance */}
        <link rel="preconnect" href="https://cdn.jsdelivr.net" crossOrigin="anonymous" />
      </Head>
      <body>
        <Main />
        <NextScript />
        {/* Prevent Monaco Editor from auto-initializing */}
        <script dangerouslySetInnerHTML={{
          __html: `
            window.MonacoEnvironment = {
              getWorkerUrl: function (moduleId, label) {
                return '/monaco-editor-worker.js';
              },
            };
          `
        }} />
      </body>
    </Html>
  );
} 