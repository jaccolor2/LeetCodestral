import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        accent: "var(--accent)",
        secondary: "var(--secondary)",
      },
      keyframes: {
        shine: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' }
        },
        'wave-fast': {
          '0%': { transform: 'translateX(-100%)' },
          '50%': { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(100%)' }
        },
        'wave-medium': {
          '0%': { transform: 'translateX(-100%)' },
          '50%': { transform: 'translateX(20%)' },
          '100%': { transform: 'translateX(100%)' }
        },
        'wave-slow': {
          '0%': { transform: 'translateX(-100%)' },
          '50%': { transform: 'translateX(-30%)' },
          '100%': { transform: 'translateX(100%)' }
        }
      },
      animation: {
        'shine': 'shine 1.5s ease-in-out infinite',
        'wave-fast': 'wave-fast 2s cubic-bezier(0.4, 0, 0.2, 1) infinite',
        'wave-medium': 'wave-medium 3s cubic-bezier(0.4, 0, 0.2, 1) infinite 0.5s',
        'wave-slow': 'wave-slow 4s cubic-bezier(0.4, 0, 0.2, 1) infinite 1s'
      }
    },
  },
  plugins: [],
} satisfies Config;
