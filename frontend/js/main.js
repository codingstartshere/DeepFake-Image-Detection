import { initializeUpload } from "./upload.js";
import { predictImage } from "./api.js";


/* ---------- DOM Elements ---------- */

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewWrapper = document.getElementById("previewWrapper");
const imagePreview = document.getElementById("imagePreview");
const checkButton = document.getElementById("checkButton");

const resultValue = document.getElementById("resultValue");
const resultDescription =
  document.getElementById("resultDescription");


/* ---------- Upload Initialization ---------- */

const uploadController = initializeUpload({
  imageInput,
  uploadArea,
  uploadPrompt,
  previewWrapper,
  imagePreview,
  checkButton,
  resultValue,
  resultDescription
});


/* ---------- Check Button ---------- */

checkButton.addEventListener("click", async () => {
  const selectedFile =
    uploadController.getSelectedFile();

  if (!selectedFile) {
    return;
  }

  /*
   * ---------------------------------------------------------
   * TEMPORARY DEMONSTRATION
   * ---------------------------------------------------------
   *
   * The backend/model has not been connected yet.
   *
   * This reproduces the behavior of your original UI.
   *
   * Later, replace this section with the actual
   * predictImage(selectedFile) API call.
   */

  resultValue.textContent = "REAL";
  resultValue.className = "result-value real";

  resultDescription.textContent =
    "Sample prediction — connect your detection model here.";


  /*
   * ---------------------------------------------------------
   * FUTURE BACKEND IMPLEMENTATION
   * ---------------------------------------------------------
   *
   * When the FastAPI backend is ready, replace the
   * temporary demonstration above with something like:
   *
   * try {
   *
   *   resultDescription.textContent =
   *     "Analyzing image...";
   *
   *   const result =
   *     await predictImage(selectedFile);
   *
   *   if (result.prediction === "REAL") {
   *     resultValue.textContent = "REAL";
   *     resultValue.className =
   *       "result-value real";
   *   } else {
   *     resultValue.textContent = "FAKE";
   *     resultValue.className =
   *       "result-value fake";
   *   }
   *
   *   resultDescription.textContent =
   *     `Confidence: ${result.confidence}%`;
   *
   * } catch (error) {
   *
   *   console.error(error);
   *
   *   resultValue.textContent = "ERROR";
   *   resultValue.className =
   *     "result-value placeholder";
   *
   *   resultDescription.textContent =
   *     "Unable to analyze the image.";
   *
   * }
   */
});