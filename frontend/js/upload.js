/**
 * Image upload and preview functionality.
 */

/**
 * Initialize the image upload functionality.
 *
 * @param {Object} elements - DOM elements used by the uploader.
 * @returns {Object} Upload controller.
 */
export function initializeUpload(elements) {
  const {
    imageInput,
    uploadArea,
    uploadPrompt,
    previewWrapper,
    imagePreview,
    checkButton,
    resultValue,
    resultDescription
  } = elements;

  let selectedFile = null;

  /**
   * Handle image selection.
   */
  function handleImageSelection() {
    const file = imageInput.files[0];

    if (!file) {
      resetInterface();
      return;
    }

    if (!file.type.startsWith("image/")) {
      resetInterface();
      return;
    }

    selectedFile = file;

    const imageUrl = URL.createObjectURL(file);

    imagePreview.src = imageUrl;

    uploadPrompt.style.display = "none";
    previewWrapper.classList.add("visible");
    uploadArea.classList.add("has-image");

    checkButton.disabled = false;

    // Reset previous result when a new image is selected.
    resultValue.textContent = "—";
    resultValue.className = "result-value placeholder";

    resultDescription.textContent = "Ready for analysis.";
  }

  /**
   * Reset the upload interface.
   */
  function resetInterface() {
    selectedFile = null;

    imageInput.value = "";

    imagePreview.removeAttribute("src");

    uploadPrompt.style.display = "flex";
    previewWrapper.classList.remove("visible");
    uploadArea.classList.remove("has-image");

    checkButton.disabled = true;

    resultValue.textContent = "—";
    resultValue.className = "result-value placeholder";

    resultDescription.textContent =
      "Upload an image to begin.";
  }

  /**
   * Return the currently selected image file.
   *
   * @returns {File|null}
   */
  function getSelectedFile() {
    return selectedFile;
  }

  /**
   * Handle file input changes.
   */
  imageInput.addEventListener(
    "change",
    handleImageSelection
  );

  /**
   * Allow clicking the upload area to select another image.
   *
   * Avoid triggering the file picker when clicking
   * the preview or browse button.
   */
  uploadArea.addEventListener("click", (event) => {
    if (
      event.target === imagePreview ||
      event.target.closest(".browse-button")
    ) {
      return;
    }

    imageInput.click();
  });

  return {
    getSelectedFile,
    resetInterface
  };
}