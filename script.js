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
  //Formulas to get effect attributes. TODO: fiddle with these till they sound right
  // let playbackRate = (rightArmAngle + 90) / 90;
  // let decay = leftArmAngle;
  // let distort = rightFootDiff;
  // let delay = leftFootDiff;
  // let feedback = leftFootDiff;
  // let pitch = headDiff;

  // if(reverbFlag.value && leftArmAngle !== -1){
  //   let decay = leftArmAngle;
  //   reverb.decay = parseFloat(clamp(decay, 50, 350));
  //   console.log("Decay: " + decay);
  // }

  if (autoWahFlag.value && leftArmAngle !== -1) {
    let octaves = (leftArmAngle + 90) / 90;
    autoWah.octaves = parseFloat(clamp(octaves, 0, 2));
    console.log("Octaves: " + octaves);
  }

  if (feedbackDelayFlag.value && rightArmAngle !== -1) {
    let feedback = (rightArmAngle + 90) / 180;
    let delay = (rightArmAngle + 90) / 180;
    feedbackDelay.delayTime.value = parseFloat(clamp(delay, 0.5, 1)); //seconds, any value
    feedbackDelay.feedback.value = parseFloat(clamp(feedback, 0, 0.5)); //between [0,1]
    console.log("Feedback: " + feedback);
    console.log("Delay: " + delay);
  }

  if (pitchShiftFlag.value && leftFootDiff !== -1) {
    let pitch = Math.abs(leftFootDiff * 60);
    pitchShift.pitch = parseFloat(clamp(pitch, 0, 12)); //half step increments, [0,12]
    console.log("Pitch: " + pitch);
  }

  if (playbackRateFlag.value && headDiff !== -1) {
    let playbackRate = Math.abs(headDiff * 10) + 0.3;
    audioElement.playbackRate = parseFloat(clamp(playbackRate, 0.75, 1.25)); // [.2, 1.8]
    console.log("PlaybackRate: " + playbackRate);
  }

  // if(distortionFlag.value && rightFootDiff !== -1){
  //   let distort = Math.abs(rightFootDiff * 5000);
  //   distortion.distortion = parseFloat(clamp(distort, 0, 500)); //between [0,1]
  //   console.log("Distort: " + distort);
  // }
  if (autoFilterFlag && rightFootDiff !== -1) {
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
Tone.connect(player, Tone.getDestination());
let pitchShift = new Tone.PitchShift().toDestination();
// let reverb = new Tone.Reverb().toDestination();
let autoWah = new Tone.AutoWah().toDestination();
// let distortion = new Tone.Distortion().toDestination();
let autoFilter = new Tone.Vibrato({
  frequency: 500,
  depth: 1,
}).toDestination();
let feedbackDelay = new Tone.FeedbackDelay().toDestination();
var playbackRateFlag = { value: false };
var pitchShiftFlag = { value: false };
// var reverbFlag = {value: false};
var autoWahFlag = { value: false };
// var distortionFlag = {value: false};
var autoFilterFlag = { value: false };
var feedbackDelayFlag = { value: false };

function handleFileUpload(event) {
  const file = event.target.files[0];
  //const fileUrl = 'Sean Paul, J Balvin - Contra La Pared.wav'
  const fileUrl = URL.createObjectURL(file);
  Tone.start();

  audioElement.src = fileUrl;
  setUpEffects(player);
}

function setUpEffects(tonePlayer) {
  var checkboxes = document.querySelectorAll('input[type="checkbox"]');
  let effectMap = {
    pitchShift: pitchShift,
    // reverb: reverb,
    autoWah: autoWah,
    // distortion: distortion,
    autoFilter: autoFilter,
    feedbackDelay: feedbackDelay,
  };

  let flagMap = {
    pitchShift: pitchShiftFlag,
    // reverb: reverbFlag,
    autoWah: autoWahFlag,
    // distortion: distortionFlag,
    autoFilter: autoFilterFlag,
    feedbackDelay: feedbackDelayFlag,
    playbackRate: playbackRateFlag,
  };

  // Loop through each checkbox
  Object.keys(flagMap).forEach(function (key) {
    flagMap[key].value = false;
  });

  checkboxes.forEach(function (checkbox) {
    // Check if the checkbox is checked
    if (checkbox.checked) {
      console.log(checkbox.value + " is checked.");
      if (checkbox.value !== "playbackRate") {
        Tone.connect(tonePlayer, effectMap[checkbox.value]);
      }
      flagMap[checkbox.value].value = true;
      console.log(playbackRateFlag);
      console.log(flagMap);
    }
  });
}

// Get references to UI elements
const playButton = document.getElementById("playButton");
const stopButton = document.getElementById("stopButton");
const audioFileInput = document.getElementById("audioFileInput");

// // Add event listeners to UI elements
// playButton.addEventListener("click", () => {
//   // Start playback
//   player.start();
// });

// stopButton.addEventListener("click", () => {
//   // Stop playback
//   player.stop();
// });

audioFileInput.addEventListener("change", handleFileUpload);
