import App from './App';

document.addEventListener('DOMContentLoaded', () => {
  const recordButton = document.getElementById('recordButton') as HTMLButtonElement;
  const audioPlayer = document.getElementById('audioPlayer') as HTMLAudioElement;

  console.log('Record Button:', recordButton);
  console.log('Audio Player:', audioPlayer);

  // 检查是否成功获取按钮和音频播放器元素
  if (recordButton && audioPlayer) {
    const app = new App(recordButton, audioPlayer);
    app.init();
  } else {
    console.error('Failed to load record button or audio player element.');
  }
});
