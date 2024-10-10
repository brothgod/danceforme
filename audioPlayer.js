// ----------------------------------- AUDIO PLAYER -----------------------------------------
const audioElement = document.getElementById("audioElement");
let player = Tone.getContext().createMediaElementSource(audioElement);
let pitchShift = {
  name: "pitch shift",
  effect: new Tone.PitchShift(),
  flag: false,
};
let phaser = {
  name: "phaser",
  effect: new Tone.Phaser({
    frequency: 10,
    baseFrequency: 100,
  }),
  flag: false,
};
let feedbackDelay = {
  name: "feedback delay",
  effect: new Tone.FeedbackDelay(),
  flag: false,
};
let distortion = {
  name: "distortion",
  effect: new Tone.Distortion(),
  flag: false,
};

let playbackRate = { name: "playback rate", effect: null, flag: false };
let effectMap = [pitchShift, phaser, feedbackDelay, distortion, playbackRate];

function addCheckboxes(effectMap) {
  const container = document.getElementById("checkboxes");
  effectMap.forEach((effect) => {
    let str = effect.name;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = str;
    checkbox.name = str.replace(/\s+/g, "-").toLowerCase(); // Convert spaces to hyphens and make lowercase for name attribute
    checkbox.id = str.replace(/\s+/g, "-").toLowerCase() + "-checkbox"; // Unique ID for the checkbox
    checkbox.checked = false; //true; // Initially unchecked
    effect.id = checkbox.id;

    const label = document.createElement("label");
    label.htmlFor = checkbox.id;
    label.textContent = str;

    const lineBreak = document.createElement("br");

    container.appendChild(checkbox);
    container.appendChild(label);
    container.appendChild(lineBreak);
  });
}
addCheckboxes(effectMap);

function handleFileUpload(event) {
  const file = event.target.files[0];
  const fileUrl = URL.createObjectURL(file);
  Tone.start();

  audioElement.src = fileUrl;
  setUpEffects(player);
}

function setUpEffects(tonePlayer) {
  let lastEffect = tonePlayer;
  effectMap.forEach(function (effect) {
    effect.flag = false;
    let checkbox = document.getElementById(effect.id);
    if (checkbox.checked) {
      console.debug(checkbox.value + " is checked.");
      if (effect.name !== "playback rate") {
        Tone.connect(lastEffect, effect.effect);
        lastEffect = effect.effect;
      }
      effect.flag = true;
    }
  });
  Tone.connect(lastEffect, Tone.getDestination());
}

function changeEffects() {
  Tone.disconnect(player);
  effectMap.forEach(function (effect) {
    if (effect.name !== "playback rate") Tone.disconnect(effect.effect);
  });

  setUpEffects(player);
  audioElement.playbackRate = 1;
}

document.getElementById("openSettings").addEventListener("click", function () {
  settingsPopup.classList.add("show");
});
document.getElementById("closeSettings").addEventListener("click", function () {
  settingsPopup.classList.remove("show");
  changeEffects();
});
window.addEventListener("click", function (event) {
  if (event.target == settingsPopup) {
    settingsPopup.classList.remove("show");
    changeEffects();
  }
});

// Get references to UI elements
const audioFileInput = document.getElementById("audioFileInput");
audioFileInput.addEventListener("change", handleFileUpload);
