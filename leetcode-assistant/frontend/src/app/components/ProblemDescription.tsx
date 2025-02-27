import { ReactElement, JSXElementConstructor, ReactNode, ReactPortal, Key } from 'react';
import { TestCaseBox } from './TestCaseBox';

interface Example {
  input: string;
  output?: string;
  expected_output?: string;
  explanation?: string;
}

interface ProblemDescriptionProps {
  problem: {
    id: number;
    title: string;
    description: string;
    difficulty: string;
    constraints?: string[];
    examples: Example[];
  } | null;
  code: string;
  onRunTests: () => Promise<any>;
}

export function ProblemDescription({ problem, code, onRunTests }: ProblemDescriptionProps) {
  if (!problem) {
    return <div className="text-gray-200">Loading problem...</div>;
  }

  const difficultyColor = {
    easy: 'bg-[#FF4405]/30',
    medium: 'bg-[#FF4405]/60',
    hard: 'bg-[#FF4405]'
  } as const;

  return (
    <div className="text-gray-200 h-full overflow-y-auto scrollbar-thin">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">{problem.title}</h1>
        <span className={`${difficultyColor[problem.difficulty as keyof typeof difficultyColor]} text-white px-3 py-1 rounded-full text-sm`}>
          {problem.difficulty}
        </span>
      </div>
      
      <div className="prose prose-invert">
        <p>{problem.description}</p>
        
        {problem.constraints && (
          <div className="mt-6">
            <h3 className="font-semibold mb-2">Constraints</h3>
            <ul className="list-disc pl-5">
              {problem.constraints.map((constraint: string | number | bigint | boolean | ReactElement<unknown, string | JSXElementConstructor<any>> | Iterable<ReactNode> | ReactPortal | Promise<string | number | bigint | boolean | ReactPortal | ReactElement<unknown, string | JSXElementConstructor<any>> | Iterable<ReactNode> | null | undefined> | null | undefined, index: Key | null | undefined) => (
                <li key={index}>{constraint}</li>
              ))}
            </ul>
          </div>
        )}

        {problem.examples.map((example: Example, index: number) => (
          <div key={index} className="mt-6 p-4 bg-[#FF4405]/10 rounded-lg">
            <h3 className="font-semibold mb-2">Example {index + 1}</h3>
            <div className="space-y-2">
              <div>
                <span className="font-medium">Input:</span>
                <pre className="bg-[#1A1A1A] p-2 rounded mt-1 text-white">{example.input}</pre>
              </div>
              <div>
                <span className="font-medium">Output:</span>
                <pre className="bg-[#1A1A1A] p-2 rounded mt-1 text-white">
                  {example.expected_output || example.output}
                </pre>
              </div>
              {example.explanation && (
                <div>
                  <span className="font-medium">Explanation:</span>
                  <p className="mt-1">{example.explanation}</p>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      
      <TestCaseBox 
        problemId={problem.id}
        code={code}
        onRunTests={onRunTests}
      />
    </div>
  );
} 