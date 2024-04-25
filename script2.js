function handleFileUpload(event) {
  const file = event.target.files[0];
  const fileUrl = Tone.start();
  const player = new Tone.Player().toDestination();
  //   "https://tonejs.github.io/audio/berklee/gong_1.mp3"
  // "Sean Paul, J Balvin - Contra La Pared.wav"
  // file
  // play as soon as the buffer is loaded
  player.autostart = true;
}
audioFileInput.addEventListener("change", handleFileUpload);
