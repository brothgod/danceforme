let player;

function handleFileUpload(event) {
  const file = event.target.files[0];
  const fileUrl = URL.createObjectURL(file);
  Tone.start();
  player = new Tone.Player(fileUrl).toDestination();
}

// Get references to UI elements
const playButton = document.getElementById("playButton");
const stopButton = document.getElementById("stopButton");
const playbackRateInput = document.getElementById("playbackRate");

// Add event listeners to UI elements
playButton.addEventListener("click", () => {
  // Start playback
  player.play();
});

stopButton.addEventListener("click", () => {
  // Stop playback
  player.stop();
});

// Update playback rate based on input value
playbackRateInput.addEventListener("change", function () {
  player.playbackRate = parseFloat(this.value);
});

audioFileInput.addEventListener("change", handleFileUpload);
