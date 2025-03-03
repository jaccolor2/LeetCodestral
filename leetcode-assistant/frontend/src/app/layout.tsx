import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const dynamic = 'force-dynamic';

export const metadata: Metadata = {
  title: "LeetCodestral",
  description: "Your AI-powered LeetCode assistant",
  icons: {
    icon: [
      {
        url: "/assets/mistral-logo.png",
        type: "image/png",
      },
    ],
    shortcut: ["/assets/mistral-logo.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/assets/mistral-logo.png" />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if (window.location.pathname.includes('/login')) {
                // Disable Monaco Editor on login page
                window.MonacoEnvironment = {
                  getWorkerUrl: function() { return ''; }
                };
              }
            `,
          }}
        />
        <script src="/login-fix.js" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
