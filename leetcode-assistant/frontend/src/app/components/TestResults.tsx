interface TestResult {
  description: string;
  input: string;
  expected_output: string;
  output: string;
  passed: boolean;
}

interface TestResultsProps {
  results: TestResult[];
}

export default function TestResults({ results }: TestResultsProps) {
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Test Results</h2>
      {results.map((result, index) => (
        <div 
          key={index} 
          className={`p-4 rounded-lg ${
            result.passed ? 'bg-green-100' : 'bg-red-100'
          }`}
        >
          <div className="font-semibold">{result.description}</div>
          <div className="text-sm">
            <div>Input: {result.input}</div>
            <div>Expected: {result.expected_output}</div>
            <div>Got: {result.output}</div>
          </div>
        </div>
      ))}
    </div>
  );
} 