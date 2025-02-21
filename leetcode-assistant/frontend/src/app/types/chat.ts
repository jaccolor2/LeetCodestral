export interface Message {
  role: 'user' | 'assistant';
  content: string;
  id?: number;
  timestamp?: number;
  streaming?: boolean;
} 