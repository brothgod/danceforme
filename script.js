// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ----------------------------------- WEBCAM PROCESSING -----------------------------------------

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "100%";
const videoWidth = "100%";

// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createPoseLandmarker = async (numPeople) => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
      delegate: "GPU",
    },
    runningMode: runningMode,
    numPoses: numPeople,
  });
};
const numPeople = document.getElementById("numPeople");
numPeople.addEventListener("change", function () {
  createPoseLandmarker(this.value);
});

createPoseLandmarker(numPeople.value);

/********************************************************************
  // Demo 2: Continuously grab image from webcam stream and detect it.
  ********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmaker not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "enable predictions";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "disable predictions";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true,
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastLandmarks = null;
let count = 0;
let interval = 10;
let lastVideoTime = -1;
async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO", num_poses: 2 });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    count += 1;
    lastVideoTime = video.currentTime;
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      let rightArmAngle = -1;
      let leftArmAngle = -1;
      for (const landmark of result.landmarks) {
        // drawingUtils.drawLandmarks(landmark, {
        //   radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
        // });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        rightArmAngle += calcRightArmAngle(landmark) / result.landmarks.length;
        leftArmAngle += calcLeftArmAngle(landmark) / result.landmarks.length;
      }

      let rightFootDiff = -1;
      let leftFootDiff = -1;
      let headDiff = -1;

      if (count % interval == 0) {
        if (result.landmarks.length == numPeople.value) {
          rightFootDiff = calcLandmarkDiff(result.landmarks, lastLandmarks, 28);
          leftFootDiff = calcLandmarkDiff(result.landmarks, lastLandmarks, 27);
          headDiff = calcLandmarkDiff(result.landmarks, lastLandmarks, 0);
          lastLandmarks = result.landmarks;
          count = 0;
          // console.log("Feet difference: " + rightFootDiff);
          // console.log("Head Difference: " + headDiff);
        } else {
          count--;
        }
      }

      // console.log("Right arm angle: " + rightArmAngle);
      // console.log("Left arm angle: " + leftArmAngle);
      if (count % interval == 0) {
        adjustPlayerEffects(
          rightArmAngle,
          leftArmAngle,
          rightFootDiff,
          leftFootDiff,
          headDiff
        );
      }

      canvasCtx.restore();
    });
  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function calcRightArmAngle(landmark) {
  //Returns angle between -90 and 90
  let rightShoulder = landmark[12];
  let rightElbow = landmark[14];

  let angle = angleWithXAxis(
    rightElbow.x - rightShoulder.x,
    rightElbow.y - rightShoulder.y
  );

  const positiveAngle = angle >= 0 ? angle : 360 + angle;

  return clamp(positiveAngle - 180, -90, 90);
}

function calcLeftArmAngle(landmark) {
  //Returns angle between -90 and 90
  let leftShoulder = landmark[11];
  let leftElbow = landmark[13];

  let angle = angleWithXAxis(
    leftElbow.x - leftShoulder.x,
    leftElbow.y - leftShoulder.y
  );

  return clamp(-1 * angle, -90, 90);
}

function calcLandmarkDiff(landmarks, pastLandmarks, landmarkNum) {
  //Returns x-difference of pose number
  if (pastLandmarks == null) {
    return 0;
  }
  let landmarkX = landmarks.map((landmark) => {
    return landmark[landmarkNum].x;
  });
  let pastLandmarkX = pastLandmarks.map((landmark) => {
    return landmark[landmarkNum].x;
  });
  landmarkX.sort();
  pastLandmarkX.sort();

  let differences = 0;
  for (let i = 0; i < numPeople.value; i += 1) {
    differences += landmarkX[i] - pastLandmarkX[i];
  }

  return differences / numPeople.value;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function adjustPlayerEffects(
  rightArmAngle,
  leftArmAngle,
  rightFootDiff,
  leftFootDiff,
  headDiff
) {
  if (vibrato.flag && leftArmAngle !== -1) {
    let octaves = (leftArmAngle + 90) / 90;
    //TODO: add formula
  }

  if (feedbackDelay.flag && rightArmAngle !== -1) {
    let feedback = (rightArmAngle + 90) / 180;
    let delay = (rightArmAngle + 90) / 180;
    feedbackDelay.delayTime.value = parseFloat(clamp(delay, 0.5, 1)); //seconds, any value
    feedbackDelay.feedback.value = parseFloat(clamp(feedback, 0, 0.5)); //between [0,1]
    console.log("Feedback: " + feedback);
    console.log("Delay: " + delay);
  }

  if (pitchShift.flag && leftFootDiff !== -1) {
    let pitch = Math.abs(leftFootDiff * 60);
    pitchShift.pitch = parseFloat(clamp(pitch, 0, 12)); //half step increments, [0,12]
    console.log("Pitch: " + pitch);
  }

  if (playbackRate.flag && headDiff !== -1) {
    let playbackRate = Math.abs(headDiff * 10) + 0.3;
    audioElement.playbackRate = parseFloat(clamp(playbackRate, 0.75, 1.25)); // [.2, 1.8]
    console.log("PlaybackRate: " + playbackRate);
  }

  if (phaser.flag && rightFootDiff !== -1) {
    //TODOL add phaser formula
    let baseFrequency = rightFootDiff * 10;
    // autoFilter.depth = parseFloat(clamp(baseFrequency, 0, 1));
    console.log("Frequency: " + baseFrequency);
  }

  console.log("------------------");
}
function angleWithXAxis(x, y) {
  // Calculate the angle in radians using Math.atan2
  const angleRadians = Math.atan2(y, x);

  // Convert radians to degrees
  const angleDegrees = angleRadians * (180 / Math.PI);

  return angleDegrees;
}

// // ----------------------------------- AUDIO PLAYER -----------------------------------------
const audioElement = document.getElementById("audioElement");
let player = Tone.getContext().createMediaElementSource(audioElement);
audioElement.autoplay = true;
audioElement.src = "song-files/Sean Paul, J Balvin - Contra La Pared.wav";
let pitchShift = {
  name: "pitch shift",
  effect: new Tone.PitchShift(),
  flag: false,
};
let vibrato = {
  name: "vibrato",
  effect: new Tone.Vibrato({
    frequency: 10,
  }),
  flag: false,
};
let phaser = {
  name: "phaser",
  effect: new Tone.Phaser({
    frequency: 10,
    baseFrequency: 1000,
  }),
  flag: false,
};
let feedbackDelay = {
  name: "feedback delay",
  effect: new Tone.FeedbackDelay(),
  flag: false,
};

let playbackRate = { name: "playback rate", effect: null, flag: false };
let effectMap = [pitchShift, phaser, feedbackDelay, vibrato, playbackRate];

function addCheckboxes(effectMap) {
  const container = document.getElementById("checkboxes");
  effectMap.forEach((effect) => {
    let str = effect.name;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = str;
    checkbox.name = str.replace(/\s+/g, "-").toLowerCase(); // Convert spaces to hyphens and make lowercase for name attribute
    checkbox.id = str.replace(/\s+/g, "-").toLowerCase() + "-checkbox"; // Unique ID for the checkbox
    checkbox.checked = true; // Initially unchecked
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
  //const fileUrl = 'Sean Paul, J Balvin - Contra La Pared.wav'
  const fileUrl = URL.createObjectURL(file);
  Tone.start();

  audioElement.src = fileUrl;
  setUpEffects(player);
}

function setUpEffects(tonePlayer) {
  let lastEffect = tonePlayer;
  effectMap.forEach(function (effect) {
    let checkbox = document.getElementById(effect.id);
    if (checkbox.checked) {
      console.log(checkbox.value + " is checked.");
      if (effect.name !== "playback rate") {
        Tone.connect(lastEffect, effect.effect);
        lastEffect = effect.effect;
      }
      effect.flag = true;
      console.log(effectMap);
    }
  });
  Tone.connect(lastEffect, Tone.getDestination());
}
setUpEffects(player);

// Get references to UI elements
const audioFileInput = document.getElementById("audioFileInput");
audioFileInput.addEventListener("change", handleFileUpload);
