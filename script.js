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

const demosSection = document.getElementById("demos");

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
  demosSection.classList.remove("invisible");
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
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
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

      let rightArmAngle = 0;
      let leftArmAngle = 0;
      let legDifference = 0;
      let hipAngle = 0;
      for (const landmark of result.landmarks) {
        // drawingUtils.drawLandmarks(landmark, {
        //   radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
        // });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        rightArmAngle +=
          calculateRightArmAngle(landmark) / result.landmarks.length;
        leftArmAngle +=
          calculateLeftArmAngle(landmark) / result.landmarks.length;
      }
      console.log("right arm" + rightArmAngle);
      console.log("left arm" + leftArmAngle);
      adjustPlayerEffects(rightArmAngle, leftArmAngle);

      canvasCtx.restore();
    });
  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function calculateRightArmAngle(landmark) {
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

function calculateLeftArmAngle(landmark) {
  //Returns angle between -90 and 90
  let leftShoulder = landmark[11];
  let leftElbow = landmark[13];

  let angle = angleWithXAxis(
    leftElbow.x - leftShoulder.x,
    leftElbow.y - leftShoulder.y
  );

  return clamp(-1 * angle, -90, 90);
}

function calculateFeetDifference(landmarks, pastLandmarks) {
  //Returns the feet movement difference
  let rightFoot;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function adjustPlayerEffects(rightArmAngle, leftArmAngle) {
  let playback_rate = (rightArmAngle + 90) / 90;

  if (typeof player !== "undefined")
    player.playbackRate = parseFloat(Math.max(playback_rate, 0.2));
}

function angleWithXAxis(x, y) {
  // Calculate the angle in radians using Math.atan2
  const angleRadians = Math.atan2(y, x);

  // Convert radians to degrees
  const angleDegrees = angleRadians * (180 / Math.PI);

  return angleDegrees;
}

// // ----------------------------------- AUDIO PLAYER -----------------------------------------

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
const audioFileInput = document.getElementById("audioFileInput");

// Add event listeners to UI elements
playButton.addEventListener("click", () => {
  // Start playback
  player.start();
});

stopButton.addEventListener("click", () => {
  // Stop playback
  player.stop();
});

// Update playback rate based on input value
playbackRateInput.addEventListener("input", function () {
  player.playbackRate = parseFloat(this.value);
  console.log(this.value);
});

audioFileInput.addEventListener("change", handleFileUpload);
