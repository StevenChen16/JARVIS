import Recorder from 'recorder-js';
import axios from 'axios';

export default class App {
  private recordButton: HTMLButtonElement;
  private audioPlayer: HTMLAudioElement;
  private recorder: Recorder;
  private audioContext: AudioContext;

  constructor(recordButton: HTMLButtonElement, audioPlayer: HTMLAudioElement) {
    this.recordButton = recordButton;
    this.audioPlayer = audioPlayer;
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.recorder = new Recorder(this.audioContext);
  }

  public async init() {
    await this.setupRecorder();
    this.recordButton.addEventListener('click', () => this.toggleRecording());
  }

  private async setupRecorder() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.recorder.init(stream);
    } catch (error) {
      console.error('Error initializing recorder:', error);
    }
  }

  private async toggleRecording() {
    console.log('Button clicked');
    if (this.recorder.isRecording()) {
      await this.stopRecording();
    } else {
      await this.startRecording();
    }
  }
  

  private async startRecording() {
    this.recordButton.textContent = 'Stop Recording';
    await this.recorder.start();
  }

  private async stopRecording() {
    this.recordButton.disabled = true;
    this.recordButton.textContent = 'Recording...';

    const { blob } = await this.recorder.stop();
    await this.sendAudio(blob);
  }

  private async sendAudio(audioBlob: Blob) {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');

      const response = await axios.post('http://127.0.0.1:10000/generate_stream', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });

      const audioUrl = URL.createObjectURL(response.data);
      this.playAudio(audioUrl);
    } catch (error) {
      console.error('Error sending audio:', error);
    }
  }

  private playAudio(audioUrl: string) {
    this.audioPlayer.src = audioUrl;
    this.audioPlayer.play();
    this.audioPlayer.onended = () => {
      this.recordButton.disabled = false;
      this.recordButton.textContent = 'Start Recording';
    };
  }

  private async processAudio(audioPath: string): Promise<string> {
    // Implement the processAudio method to extract audio tokens and construct input prompt
    // This is a placeholder implementation
    return "<|begin_of_audio|><|audio_0|><|end_of_audio|>";
  }

  private async inferenceFn(audioTokens: string, temperature = 0.2, top_p = 0.8, max_new_token = 2000) {
    // Implement the inferenceFn method to send request to backend and process response
    // This is a placeholder implementation
    console.log("Inference function called with tokens:", audioTokens);
  }
}
