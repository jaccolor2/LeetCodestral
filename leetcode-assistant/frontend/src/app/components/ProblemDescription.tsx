interface ProblemDescriptionProps {
  problem: {
    id: number;
    title: string;
    difficulty: string;
    description: string;
    examples: { input: string; output: string; explanation?: string }[];
    constraints?: string[];
  };
}

export function ProblemDescription({ problem }: ProblemDescriptionProps) {
  const difficultyColor = {
    Easy: 'bg-green-500',
    Medium: 'bg-yellow-500',
    Hard: 'bg-red-500'
  } as const;

  return (
    <div className="text-gray-200 h-full overflow-y-auto p-4">
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
              {problem.constraints.map((constraint, index) => (
                <li key={index}>{constraint}</li>
              ))}
            </ul>
          </div>
        )}

        {problem.examples.map((example, index) => (
          <div key={index} className="mt-6 p-4 bg-gray-700 rounded-lg">
            <h3 className="font-semibold mb-2">Example {index + 1}</h3>
            <div className="space-y-2">
              <div>
                <span className="font-medium">Input:</span>
                <pre className="bg-gray-800 p-2 rounded mt-1">{example.input}</pre>
              </div>
              <div>
                <span className="font-medium">Output:</span>
                <pre className="bg-gray-800 p-2 rounded mt-1">{example.output}</pre>
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
    </div>
  );
} 