import React, { useState, useRef, useEffect } from 'react';
import { Mic, Send, Trash2, Volume2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';

// Types
interface Message {
  role: 'user' | 'assistant';
  content: string | { path: string; type?: string };
}

interface ChatSettings {
  temperature: number;
  topP: number;
  maxNewTokens: number;
}

const VoiceChatApp = () => {
  // States
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMode, setInputMode] = useState<'audio' | 'text'>('audio');
  const [textInput, setTextInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    temperature: 0.2,
    topP: 0.8,
    maxNewTokens: 2000,
  });
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Audio Recording Functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      audioStream?.getTracks().forEach(track => track.stop());
      setIsRecording(false);

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        handleSubmit(audioBlob);
      };
    }
  };

  // API Communication
  const handleSubmit = async (audioBlob?: Blob) => {
    const formData = new FormData();
    
    if (inputMode === 'audio' && audioBlob) {
      formData.append('audio', audioBlob);
    } else {
      formData.append('text', textInput);
    }

    formData.append('settings', JSON.stringify(settings));

    try {
      setIsStreaming(true);
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const data = JSON.parse(chunk);

        if (data.audio) {
          // Handle streaming audio
          if (audioRef.current) {
            audioRef.current.src = URL.createObjectURL(
              new Blob([data.audio], { type: 'audio/wav' })
            );
            audioRef.current.play();
          }
        }
        
        if (data.text) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: data.text
          }]);
        }
      }
    } catch (error) {
      console.error('Error in chat:', error);
    } finally {
      setIsStreaming(false);
      setTextInput('');
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Voice Chat</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 mb-4">
            <div className="flex flex-col gap-2">
              <Label>Temperature</Label>
              <Slider 
                value={[settings.temperature]}
                min={0}
                max={1}
                step={0.1}
                onValueChange={([value]) => setSettings(prev => ({ ...prev, temperature: value }))}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label>Top P</Label>
              <Slider 
                value={[settings.topP]}
                min={0}
                max={1}
                step={0.1}
                onValueChange={([value]) => setSettings(prev => ({ ...prev, topP: value }))}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label>Max Tokens</Label>
              <Input 
                type="number" 
                value={settings.maxNewTokens}
                onChange={(e) => setSettings(prev => ({ ...prev, maxNewTokens: parseInt(e.target.value) }))}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex-1 overflow-y-auto bg-white rounded-lg p-4 space-y-4">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-[70%] p-3 rounded-lg ${
              message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-100'
            }`}>
              {typeof message.content === 'string' ? (
                <p>{message.content}</p>
              ) : (
                <audio controls src={message.content.path} />
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4">
        <RadioGroup 
          value={inputMode} 
          onValueChange={(value: 'audio' | 'text') => setInputMode(value)}
          className="flex gap-4 mb-4"
        >
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="audio" id="audio" />
            <Label htmlFor="audio">Audio</Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="text" id="text" />
            <Label htmlFor="text">Text</Label>
          </div>
        </RadioGroup>

        <div className="flex gap-2">
          {inputMode === 'audio' ? (
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-12 h-12 rounded-full ${isRecording ? 'bg-red-500' : 'bg-blue-500'}`}
            >
              <Mic className={isRecording ? 'animate-pulse' : ''} />
            </Button>
          ) : (
            <>
              <Input
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Type your message..."
                className="flex-1"
              />
              <Button onClick={() => handleSubmit()} disabled={isStreaming}>
                <Send />
              </Button>
            </>
          )}
          <Button variant="outline" onClick={() => setMessages([])}>
            <Trash2 />
          </Button>
        </div>
      </div>

      <audio ref={audioRef} className="hidden" />
    </div>
  );
};

export default VoiceChatApp;