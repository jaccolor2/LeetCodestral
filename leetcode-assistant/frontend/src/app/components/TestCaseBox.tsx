import { useState } from 'react';

interface TestCase {
  input: string;
  expected_output: string;
  description: string;
  passed?: boolean;
  output?: string;
}

interface TestCaseBoxProps {
  problemId: number;
  code: string;
  onRunTests: () => Promise<{ results: TestCase[] }>;
}

export function TestCaseBox({ problemId, code, onRunTests }: TestCaseBoxProps) {
  const [testCases, setTestCases] = useState<TestCase[]>([]);
  const [loading, setLoading] = useState(false);

  const handleRunTests = async () => {
    setLoading(true);
    try {
      const response = await onRunTests();
      setTestCases(response.results);
    } catch (error) {
      console.error('Error running tests:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-8 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Test Cases</h2>
        <button
          onClick={handleRunTests}
          disabled={loading}
          className="bg-[#FF4405] hover:bg-[#FF4405]/80 text-white px-4 py-2 rounded disabled:opacity-50"
        >
          {loading ? 'Running Tests...' : 'Run Tests'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {testCases.length > 0 ? (
          <div className="space-y-4">
            {testCases.map((test, index) => (
              <div key={index} className={`p-4 rounded-lg ${test.passed ? 'bg-green-900/30' : 'bg-red-900/30'}`}>
                <div className="flex items-center justify-between">
                  <div className="font-medium">{test.description}</div>
                  <div className={`text-sm ${test.passed ? 'text-green-400' : 'text-red-400'}`}>
                    {test.passed ? 'Passed' : 'Failed'}
                  </div>
                </div>
                <div className="mt-2 space-y-1">
                  <div className="text-sm">
                    <span className="text-gray-500">Input:</span>
                    <span className="text-white ml-2">{test.input}</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-500">Expected:</span>
                    <span className="text-white ml-2">{test.expected_output}</span>
                  </div>
                  {test.output && (
                    <div className="text-sm">
                      <span className="text-gray-500">Output:</span>
                      <span className={`ml-2 ${test.passed ? 'text-green-400' : 'text-red-400'}`}>
                        {test.output}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-400">No test cases run yet</div>
        )}
      </div>
    </div>
  );
} 