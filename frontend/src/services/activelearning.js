const API_URL = "http://localhost:8000";

const activeLearnAPI = {
  async initializeProject({
    project_name,
    model_type,
    num_classes,
    val_split,
    initial_labeled_ratio,
    sampling_strategy,
    batch_size,
    epochs,
    learning_rate,
  }) {
    const response = await fetch(`${API_URL}/init`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_name,
        model_type,
        num_classes,
        val_split,
        initial_labeled_ratio,
        sampling_strategy,
        batch_size,
        epochs,
        learning_rate,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        error.detail || `Failed to initialize project: ${response.statusText}`,
      );
    }

    return await response.json();
  },

  async getBatch(strategy, batchSize) {
    const response = await fetch(`${API_URL}/get-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy: strategy,
        batch_size: parseInt(batchSize),
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get batch");
    }
    return response.json();
  },

  async uploadData(files, valSplit, initialLabeledRatio) {
    const formData = new FormData();
    for (let file of files) {
      formData.append("files", file);
    }
    if (valSplit !== undefined) {
      formData.append("val_split", valSplit);
    }
    if (initialLabeledRatio !== undefined) {
      formData.append("initial_labeled_ratio", initialLabeledRatio);
    }

    const response = await fetch(`${API_URL}/upload-data`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload data: ${response.statusText}`);
    }

    return await response.json();
  },

  async submitLabel(imageId, label) {
    const response = await fetch(`${API_URL}/submit-label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: imageId,
        label: label,
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to submit label");
    }
    const result = await response.json();
    return result; // Now includes batch_complete and is_training flags
  },

  async trainModel(epochs = 5, batchSize = 16, learningRate = 0.001) {
    const response = await fetch(`${API_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      }),
    });
    return response.json();
  },

  async startEpisode(epochs = 5, batchSize = 16, learningRate = 0.001) {
    const response = await fetch(`${API_URL}/train-episode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        error.detail || `Failed to start episode: ${response.statusText}`,
      );
    }

    const started = await response.json();
    if (started.status !== "started") {
      return started;
    }

    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const res = await fetch(`${API_URL}/training-status`);
          if (!res.ok) {
            setTimeout(poll, 2000);
            return;
          }
          const s = await res.json();
          if (s.status === "running") {
            setTimeout(poll, 2000);
          } else if (s.status === "complete") {
            resolve(s.result);
          } else if (s.status === "error") {
            reject(new Error(s.error || "Training failed"));
          } else {
            setTimeout(poll, 2000);
          }
        } catch (e) {
          setTimeout(poll, 2000);
        }
      };
      poll();
    });
  },

  async recoverBatch(strategy = "random", batchSize = 16) {
    try {
      console.log(
        `Attempting to recover batch with strategy: ${strategy}, size: ${batchSize}`,
      );

      // Try to get a new batch with a simpler strategy
      const response = await fetch(`${API_URL}/recover-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy: strategy,
          batch_size: batchSize,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to recover batch");
      }

      return await response.json();
    } catch (error) {
      console.error("Batch recovery failed:", error);
      throw error;
    }
  },
  // Add to activelearning.js
  async getBatchSafe(strategy = "least_confidence", batchSize = 16) {
    try {
      // Ensure valid parameters
      const validStrategy = strategy || "least_confidence";
      const validBatchSize = parseInt(batchSize) || 16;

      console.log(
        `Getting batch with strategy: ${validStrategy}, size: ${validBatchSize}`,
      );

      const response = await fetch(`${API_URL}/get-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy: validStrategy,
          batch_size: validBatchSize,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get batch");
      }

      return response.json();
    } catch (error) {
      console.error("getBatchSafe error:", error);

      // Try with fallback random strategy
      try {
        console.log("Attempting fallback with random strategy");
        const response = await fetch(`${API_URL}/get-batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            strategy: "random",
            batch_size: Math.min(parseInt(batchSize) || 16, 10),
          }),
        });

        if (!response.ok) {
          throw new Error("Fallback also failed");
        }

        return response.json();
      } catch (fallbackError) {
        console.error("Fallback failed:", fallbackError);
        throw new Error(
          "Failed to get batch with both regular and fallback strategies",
        );
      }
    }
  },

  async getStatus() {
    const response = await fetch(`${API_URL}/status`);
    return response.json();
  },

  async getMetrics() {
    const response = await fetch(`${API_URL}/metrics`);
    return response.json();
  },

  async getEpisodeHistory() {
    const response = await fetch(`${API_URL}/episode-history`);
    if (!response.ok) {
      throw new Error(`Failed to get episode history: ${response.statusText}`);
    }
    return await response.json();
  },

  getImageUrl(imageId) {
    return `${API_URL}/image/${imageId}`;
  },

  async importModel(file) {
    const formData = new FormData();
    formData.append("uploaded_file", file); // Match the parameter name in FastAPI

    const response = await fetch(`${API_URL}/import-model`, {
      method: "POST",
      body: formData, // Don't set Content-Type header, let browser set it with boundary
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to import model");
    }

    return response.json();
  },

  async getValidationStatus() {
    const response = await fetch(`${API_URL}/validation-status`);
    if (!response.ok) {
      throw new Error(
        `Failed to get validation status: ${response.statusText}`,
      );
    }
    return await response.json();
  },

  async startAutomatedTraining(config) {
    const newConfig = {
      epochs: config.epochs,
      batch_size: config.batch_size,
      sampling_strategy: config.sampling_strategy,
      learning_rate: 0.001,
    };

    const response = await fetch(`${API_URL}/start-automated-training`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json", // This is important
      },
      body: JSON.stringify(newConfig),
    });

    if (!response.ok) {
      const error = await response.json();
      console.log(error);
      throw new Error(
        error.detail ||
          `Failed to start automated training: ${JSON.stringify(error)}`,
      );
    }
    return response.json();
  },

  async stopAutomatedTraining() {
    const response = await fetch(`${API_URL}/stop-automated-training`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(
        `Failed to stop automated training: ${JSON.stringify(
          response.statusText,
        )}`,
      );
    }
    return response.json();
  },

  async getAutomatedTrainingStatus() {
    const response = await fetch(`${API_URL}/automated-training-status`);
    if (!response.ok) {
      throw new Error(
        `Failed to get automated training status: ${response.statusText}`,
      );
    }
    return response.json();
  },

  async getNextBatch(strategy = "least_confidence", batchSize = 16) {
    const response = await fetch(`${API_URL}/get-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy, batch_size: batchSize }),
    });
    if (!response.ok) {
      throw new Error(`Failed to get next batch: ${response.statusText}`);
    }
    return response.json();
  },

  async resetTrainingState() {
    const response = await fetch(`${API_URL}/reset-training-state`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to reset training state: ${response.statusText}`);
    }
    return response.json();
  },

  configureLRScheduler: async (config) => {
    const response = await fetch(`${API_URL}/configure-lr-scheduler`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return response.json();
  },

  async getLabelHint(imageId) {
    try {
      const response = await fetch(`${API_URL}/label-hint/${imageId}`);
      if (!response.ok) return { available: false };
      return response.json();
    } catch {
      return { available: false };
    }
  },

  async getLabelHint(imageId) {
    try {
      const response = await fetch(`${API_URL}/label-hint/${imageId}`);
      if (!response.ok) return { available: false };
      return response.json();
    } catch {
      return { available: false };
    }
  },

  getLRSchedulerStatus: async () => {
    try {
      const response = await fetch(`${API_URL}/lr-scheduler-status`);
      if (!response.ok) {
        console.warn("Failed to get LR scheduler status");
        return null;
      }
      return response.json();
    } catch (error) {
      console.error("Error getting LR scheduler status:", error);
      return null;
    }
  },

  saveCheckpoint: async () => {
    const response = await fetch(`${API_URL}/save-checkpoint`, {
      method: "POST",
    });
    return response.json();
  },

  loadCheckpoint: async (checkpointId) => {
    const response = await fetch(`${API_URL}/load-checkpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint_path: checkpointId }),
    });
    return response.json();
  },

  listCheckpoints: async () => {
    const response = await fetch(`${API_URL}/list-checkpoints`);
    return response.json();
  },

  async importPretrainedModel(file, modelType, numClasses, projectName) {
    const formData = new FormData();
    formData.append("uploaded_file", file);
    formData.append("model_type", modelType); // Can now be "custom"
    formData.append("num_classes", numClasses.toString());
    formData.append("project_name", projectName);

    const response = await fetch(`${API_URL}/import-pretrained-model`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to import pre-trained model");
    }

    return await response.json();
  },

  async verifyModelCompatibility(file) {
    const formData = new FormData();
    formData.append("uploaded_file", file);

    const response = await fetch(`${API_URL}/verify-model-compatibility`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to verify model compatibility");
    }

    return await response.json();
  },

  async verifyCustomModel(file) {
    const formData = new FormData();
    formData.append("uploaded_file", file);

    const response = await fetch(`${API_URL}/verify-custom-model`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to verify custom model");
    }

    return await response.json();
  },

  async adaptPretrainedModel(freezeLayers, adaptationType) {
    const formData = new FormData();
    formData.append("freeze_layers", freezeLayers);
    formData.append("adaptation_type", adaptationType);

    const response = await fetch(`${API_URL}/adapt-pretrained-model`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to adapt model");
    }

    return await response.json();
  },

  async verifyModelCompatibility(file) {
    const formData = new FormData();
    formData.append("uploaded_file", file);

    const response = await fetch(`${API_URL}/verify-model-compatibility`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to verify model compatibility");
    }

    return await response.json();
  },

  async uploadCSVPaths(csvFile, delimiter = ",", valSplit, initialLabeledRatio) {
    const formData = new FormData();

    formData.append("file", csvFile);   // FIXED
    formData.append("delimiter", delimiter);

    if (valSplit !== undefined) {
      formData.append("val_split", valSplit);
    }
    if (initialLabeledRatio !== undefined) {
      formData.append("initial_labeled_ratio", initialLabeledRatio);
    }

    const response = await fetch(`${API_URL}/upload-csv-paths`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        typeof error.detail === "string"
          ? error.detail
          : JSON.stringify(error.detail)
      );
    }

    return await response.json();
  },

  async uploadCombinedWithLabels(
    files,
    labelColumn = "label",
    valSplit,
    initialLabeledRatio,
  ) {
    const formData = new FormData();

    // Add all files to the form data
    files.forEach((file) => {
      formData.append("files", file);
    });

    formData.append("label_column", labelColumn);

    if (valSplit !== undefined) {
      formData.append("val_split", valSplit);
    }
    if (initialLabeledRatio !== undefined) {
      formData.append("initial_labeled_ratio", initialLabeledRatio);
    }

    const response = await fetch(`${API_URL}/upload-combined-with-labels`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        error.detail ||
          `Failed to upload files with labels: ${response.statusText}`,
      );
    }

    return await response.json();
  },
  // Add this method to activeLearnAPI.js

  // In activelearning.js - Enhanced version of uploadCSVPathsWithLabels
  // Updated uploadCSVPathsWithLabels method in activeLearnAPI
  uploadCSVPathsWithLabels: async (
    csvFile,
    labelColumn,
    delimiter = ",",
    valSplit = 0.2,
    initialLabeledRatio = 0.4,
    expectedLabelMapping = null, // Add this parameter
  ) => {
    const formData = new FormData();
    formData.append("csv_file", csvFile);
    formData.append("label_column", labelColumn);
    formData.append("delimiter", delimiter);
    formData.append("val_split", valSplit.toString());
    formData.append("initial_labeled_ratio", initialLabeledRatio.toString());

    // Pass the expected label mapping if provided
    if (expectedLabelMapping) {
      console.log("Sending expected label mapping:", expectedLabelMapping);
      formData.append(
        "expected_label_mapping",
        JSON.stringify(expectedLabelMapping),
      );
    }

    const response = await fetch(`${API_URL}/upload-csv-paths-with-labels`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to upload CSV with labels");
    }

    return response.json();
  },

  // Also add this debug method to help troubleshoot
  async debugCSVFile(csvFile) {
    const formData = new FormData();
    formData.append("file", csvFile);

    const response = await fetch(`${API_URL}/debug-csv-file`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to debug CSV file: ${response.statusText}`);
    }

    return await response.json();
  },

  async exportProject() {
    const response = await fetch(`${API_URL}/export-project`, {
      method: "GET",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to export project");
    }
    return response.blob();
  },

  async cleanupProject() {
    const response = await fetch(`${API_URL}/cleanup-project`, {
      method: "POST",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to clean up project");
    }
    return response.json();
  },




  async importProject(file) {
    const formData = new FormData();
    formData.append("uploaded_file", file);

    const response = await fetch(`${API_URL}/import-project`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to import project");
    }

    return response.json();
  },

  async updateProjectLabels(labels) {
    const response = await fetch(`${API_URL}/update-project-labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels: labels }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to update project labels");
    }

    return response.json();
  },

  async evaluateModel(numSamples = 10) {
    const response = await fetch(
      `${API_URL}/evaluate-model?num_samples=${numSamples}`,
    );
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to evaluate model");
    }
    return response.json();
  },

  // Get evaluation batch for display
  async getEvaluationBatch(numSamples = 10) {
    const response = await fetch(
      `${API_URL}/evaluation-batch?num_samples=${numSamples}`,
    );
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get evaluation batch");
    }
    return response.json();
  },

  // Continue training after evaluation screen
  async continueFromEvaluation() {
    const response = await fetch(`${API_URL}/continue-from-evaluation`, {
      method: "POST",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to continue from evaluation");
    }
    return response.json();
  },
};

export default activeLearnAPI;
