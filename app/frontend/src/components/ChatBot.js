import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './ChatBot.css';
import { API_URL } from '../config';


const ChatBot = ({ activeConfigId }) => {
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const chatHistoryRef = useRef(null);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  useEffect(() => {
    setChatHistory([]);
  }, [activeConfigId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const newChatHistory = [...chatHistory, { sender: 'user', message: question }];
    setChatHistory(newChatHistory);
    setQuestion('');

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        question,
        app_config_id: activeConfigId,
      });
      const answer = response.data.answer;
      setChatHistory([...newChatHistory, { sender: 'bot', message: answer }]);
    } catch (error) {
      console.error('Error asking question:', error);
      setChatHistory([...newChatHistory, { sender: 'bot', message: "I'm sorry, but I encountered an error while processing your request." }]);
    }
  };

  return (
    <div className="chatbot">
      <div className="chatbot-title">Chat Assistant</div>
      <div className="chat-history" ref={chatHistoryRef}>
        {chatHistory.map((chat, index) => (
          <div key={index} className={`chat-message ${chat.sender}`}>
            {chat.sender === 'bot' ? (
              <div className="bot-markdown">
                <ReactMarkdown>{chat.message}</ReactMarkdown>
              </div>
            ) : (
              <p>{chat.message}</p>
            )}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatBot;
