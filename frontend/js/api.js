/**
 * API communication for the deepfake detection backend.
 */

/**
 * Send an image to the backend for prediction.
 *
 * @param {File} file - Image file to analyze.
 * @returns {Promise<Object>} Prediction response.
 */
export async function predictImage(file) {
  if (!file) {
    throw new Error("No image selected.");
  }

  const formData = new FormData();

  formData.append("image", file);

  const response = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error(
      `Prediction request failed with status ${response.status}.`
    );
  }

  const result = await response.json();

  return result;
}