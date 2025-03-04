# LeetCode Assistant

![Frontend Example](frontend_example.png)

An interactive coding platform that helps you learn and practice coding with AI-powered assistance.

## Features

- 🤖 AI-powered coding assistant that provides contextual help
- 💻 Interactive code editor with syntax highlighting
- ✅ Automated test case generation and validation
- 🗣️ Chat interface with conversation history
- 🔒 User authentication system
- 🔄 Multi-language support (Python and JavaScript)
- 🎯 Smart problem generation with title caching
- 🧪 Intelligent test case generation
- 💡 Encouraging success messages

## How It Works

### Problem Generation and Management

- Problems are dynamically generated using the Mistral AI API
- A title cache system prevents duplicate problems
- Each problem includes:
  - Unique title and description
  - Difficulty level
  - Example test cases


### Code Editor and Execution

- Supports both Python and JavaScript
- Automatic language detection based on code syntax
- Real-time code execution with safety limits
- Syntax highlighting and error reporting


### Test Generation and Execution

- Language-specific AI test case generation
- Automated test execution in isolated environment
- Comprehensive test results including:
  - Input values
  - Expected output
  - Actual output
  - Pass/fail status
- AI relevance checking to ensure code matches problem requirements

### Success and Validation

- Encouraging success messages when tests pass
- Technical analysis of solution strengths
- Option to proceed to next problem or continue improving


### AI Assistant Chatbot

- Context-aware assistance based on:
  - Current problem
  - User's code
  - Test results
  - Conversation history
- Content moderation for safe interactions
- Streaming responses for better UX

### Security Features

- JWT-based authentication
- Safe code execution environment
- Content moderation while chatting with the assistant


## Tech Stack

### Frontend

- Next.js 13+ with App Router
- TypeScript
- TailwindCSS
- CodeMirror for code editing
- React Markdown for chat rendering

### Backend

- FastAPI
- SQLAlchemy
- Mistral AI API
- JWT Authentication
- SQLite Database

## Getting Started

### Prerequisites

- Node.js 16+
- Python 3.8+
- Mistral AI API key
- Node.js (for JavaScript execution)
- Environment variables setup

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/jaccolor2/LeetCodestral.git
    cd LeetCodestral
    ```

2. **Set up a virtual environment for the backend**

    ```bash
    cd leetcode-assistant/backend
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install backend dependencies**

    ```bash
    pip install -r requirements.txt
    python init_db.py
    ```

4. **Create a `.env` file in the backend directory:**

    ```env
    MISTRAL_API_KEY=your_mistral_api_key
    ```

5. **Install frontend dependencies**

    ```bash
    cd ../frontend
    npm install
    ```

### Running the Application

1. **Start the backend server**

    ```bash
    cd backend
    uvicorn main:app --reload
    ```

2. **Start the frontend development server**

    ```bash
    cd frontend
    npm run dev
    ```

3. **Open [http://localhost:3000](http://localhost:3000) in your browser**


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

