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

      let rightArmAngle = 0;
      let leftArmAngle = 0;
      for (const landmark of result.landmarks) {
        // drawingUtils.drawLandmarks(landmark, {
        //   radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
        // });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        rightArmAngle += calcRightArmAngle(landmark) / result.landmarks.length;
        leftArmAngle += calcLeftArmAngle(landmark) / result.landmarks.length;
      }

      let rightFootDiff = null;
      let leftFootDiff = null;
      let headDiff = null;

      if (count % interval == 0) {
        if (result.landmarks.length == numPeople.value) {
          rightFootDiff = calcLandmarkDiff(result.landmarks, lastLandmarks, 28);
          leftFootDiff = calcLandmarkDiff(result.landmarks, lastLandmarks, 27);
          headDiff = calcLandmarkDiff(result.landmarks, lastLandmarks, 0);
          lastLandmarks = result.landmarks;
          count = 0;
          console.log("Feet difference: " + rightFootDiff);
          console.log("Head Difference: " + headDiff);
        } else {
          count--;
        }
      }

      console.log("Right arm angle: " + rightArmAngle);
      console.log("Left arm angle: " + leftArmAngle);
      adjustPlayerEffects(
        rightArmAngle,
        leftArmAngle,
        rightFootDiff,
        leftFootDiff,
        headDiff
      );

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
  //Formulas to get effect attributes. TODO: fiddle with these till they sound right
  // let playbackRate = (rightArmAngle + 90) / 90;
  // let decay = leftArmAngle;
  // let distort = rightFootDiff;
  // let delay = leftFootDiff;
  // let feedback = leftFootDiff;
  // let pitch = headDiff;

  let decay = leftArmAngle;
  let feedback = (rightArmAngle + 90) / 180;
  let delay = (rightArmAngle + 90) / 180;
  let pitch = Math.abs(leftFootDiff * 60);
  let playbackRate = Math.abs(headDiff * 10) + 0.3;
  let distort = Math.abs(rightFootDiff * 5000);

  console.log("Decay: " + decay);
  console.log("Feedback: " + feedback);
  console.log("Delay: " + delay);

  reverb.decay = parseFloat(clamp(decay, 50, 350));
  feedbackDelay.delayTime.value = parseFloat(clamp(delay, 0.5, 1)); //seconds, any value
  feedbackDelay.feedback.value = parseFloat(clamp(feedback, 0, 0.5)); //between [0,1]

  if (leftFootDiff !== null) {
    distortion.distortion = parseFloat(clamp(distort, 0, 1000)); //between [0,1]
    pitchShift.pitch = parseFloat(clamp(pitch, 0, 12)); //half step increments, [0,12]
    if (playbackRateFlag) {
      player.playbackRate = parseFloat(clamp(playbackRate, 0.75, 1.25)); // [.2, 1.8]
    }
    console.log("PlaybackRate: " + playbackRate);
    console.log("Pitch: " + pitch);
    console.log("Distort: " + distort);
  }
  console.log("------------------");
  console.log("lfd:" + leftFootDiff);
}
function angleWithXAxis(x, y) {
  // Calculate the angle in radians using Math.atan2
  const angleRadians = Math.atan2(y, x);

  // Convert radians to degrees
  const angleDegrees = angleRadians * (180 / Math.PI);

  return angleDegrees;
}

// // ----------------------------------- AUDIO PLAYER -----------------------------------------

let player = new Tone.Player();
let pitchShift = new Tone.PitchShift().toDestination();
let reverb = new Tone.Reverb().toDestination();
let distortion = new Tone.Distortion().toDestination();
let feedbackDelay = new Tone.FeedbackDelay().toDestination();
let playbackRateFlag = false;

function handleFileUpload(event) {
  const file = event.target.files[0];
  //const fileUrl = 'Sean Paul, J Balvin - Contra La Pared.wav'
  const fileUrl = URL.createObjectURL(file);
  Tone.start();
  player = new Tone.Player(fileUrl).toDestination();
  setUpEffects(player);
}

function setUpEffects(tonePlayer) {
  var checkboxes = document.querySelectorAll('input[type="checkbox"]');
  let map = {
    pitchShift: pitchShift,
    reverb: reverb,
    distortion: distortion,
    feedbackDelay: feedbackDelay,
  };

  // Loop through each checkbox
  playbackRateFlag = false;
  checkboxes.forEach(function (checkbox) {
    // Check if the checkbox is checked
    if (checkbox.checked) {
      console.log(checkbox.value + " is checked.");
      if (checkbox.value === "playbackRate") {
        playbackRateFlag = true;
      } else tonePlayer.connect(map[checkbox.value]);
    }
  });
}

// Get references to UI elements
const playButton = document.getElementById("playButton");
const stopButton = document.getElementById("stopButton");
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

audioFileInput.addEventListener("change", handleFileUpload);
