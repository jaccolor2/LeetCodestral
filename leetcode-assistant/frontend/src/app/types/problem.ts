export interface Problem {
  id: number;
  title: string;
  difficulty: string;
  description: string;
  examples: { input: string; output: string }[];
} 