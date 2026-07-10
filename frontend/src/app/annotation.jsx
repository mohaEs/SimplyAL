"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { RadioGroup, RadioGroupItem } from "../components/ui/radio-group";
import {
  AlertCircle,
  Download,
  Plus,
  Trash2,
  CheckCircle,
  Info,
} from "lucide-react";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import ImageLoader from "../components/imageLoader";
import { Alert, AlertDescription } from "../components/ui/alert";
import activeLearnAPI from "../services/activelearning";
import AutomatedTrainingControls from "@/components/automatedTrainingControls";
import { BatchProgress, ActiveLearningStatus } from "@/components/components";
import CheckpointControls from "@/components/checkpointControls";
import ProjectImport from "@/components/projectImport";
import PretrainedModelImport from "@/components/pretrainedModel";
import ModelAdaptationControls from "@/components/modelAdaptationControls";
import ModelEvaluationScreen from "@/components/modelEvaluationScreen";

const ActiveLearningUI = () => {
  // Project Configuration State
  const [projectName, setProjectName] = useState("");
  const [selectedModel, setSelectedModel] = useState("resnet50");
  const [currentImage, setCurrentImage] = useState("3559b.png");
  const [activeTab, setActiveTab] = useState("new");
  const [loadedImages, setLoadedImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [message, setMessage] = useState({ text: null, type: "error" });

  // Active Learning State
  const [samplingStrategy, setSamplingStrategy] = useState("least_confidence");
  const [batchSize, setBatchSize] = useState(16);
  const [currentBatch, setCurrentBatch] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedLabel, setSelectedLabel] = useState("");
  const [isRetraining, setIsRetraining] = useState(false);
  const [labels, setLabels] = useState([]);
  const [projectType, setProjectType] = useState("new");
  const [isBatchInProgress, setIsBatchInProgress] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isContinuing, setIsContinuing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [episodeHistory, setEpisodeHistory] = useState([]);
  const [valSplit, setValSplit] = useState(0.2);
  const [initialLabeledRatio, setInitialLabeledRatio] = useState(0.4);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [automatedStatus, setAutomatedStatus] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isProjectFullyInitialized, setIsProjectFullyInitialized] =
    useState(null);
  const [epochs, setEpochs] = useState(5);
  const [validationAccuracy, setValidationAccuracy] = useState(0);
  const [status, setStatus] = useState({
    project_name: null,
    current_episode: 0,
    labeled_count: 0,
    unlabeled_count: 0,
    validation_count: 0,
    current_batch_size: 0,
  });
  const [batchStats, setBatchStats] = useState({
    totalImages: batchSize,
    completed: 0,
    remaining: batchSize,
    accuracy: 0.85,
    timeElapsed: "00:00",
  });
  const [metrics, setMetrics] = useState({
    episodeAccuracies: [],
    currentEpochLosses: [],
  });
  const [validationStatus, setValidationStatus] = useState({
    total: 0,
    labeled: 0,
    unlabeled: 0,
    percent_labeled: 0,
  });
  const [lrConfig, setLrConfig] = useState({
    strategy: "plateau", // default strategy
    initial_lr: 0.001,
    factor: 0.1,
    patience: 5,
    min_lr: 1e-6,
  });
  const [lrHistory, setLrHistory] = useState();
  const fileDataRef = useRef(null);
  const [showEvaluationScreen, setShowEvaluationScreen] = useState(false);
  const [evaluationData, setEvaluationData] = useState(null);

  const setErrorMessage = (text) => {
    setMessage({ text, type: "error" });
  };

  const setSuccessMessage = (text) => {
    setMessage({ text, type: "success" });
  };

  const setInfoMessage = (text) => {
    setMessage({ text, type: "info" });
  };

  // Helper function to get the appropriate alert variant and icon
  const getAlertProps = (messageType) => {
    switch (messageType) {
      case "success":
        return {
          variant: "default",
          className: "border-green-200 bg-green-50 text-green-800",
          icon: CheckCircle,
        };
      case "info":
        return {
          variant: "default",
          className: "border-blue-200 bg-blue-50 text-blue-800",
          icon: Info,
        };
      case "error":
      default:
        return {
          variant: "destructive",
          className: "",
          icon: AlertCircle,
        };
    }
  };

  const getFilteredPredictions = (rawPredictions, userLabels) => {
    if (!rawPredictions || !userLabels || userLabels.length === 0) {
      console.log("No predictions or labels provided");
      return [];
    }

    console.log("=== PREDICTION FILTERING DEBUG ===");
    console.log("Raw predictions count:", rawPredictions.length);
    console.log("User labels:", userLabels);

    // Take only the first N predictions matching our label count
    const relevantPredictions = rawPredictions.slice(0, userLabels.length);
    console.log(
      "Relevant predictions (first",
      userLabels.length,
      "):",
      relevantPredictions,
    );

    // Map to use actual label names with EXPLICIT confidence preservation
    const mapped = relevantPredictions.map((pred, idx) => {
      const mappedPred = {
        label: userLabels[idx], // Use actual label name
        confidence: Number(pred.confidence), // Ensure it's a number
        labelIndex: idx,
        originalLabel: pred.label,
      };
      console.log(
        `Mapping ${idx}: ${pred.label} (${pred.confidence}) -> ${mappedPred.label} (${mappedPred.confidence})`,
      );
      return mappedPred;
    });

    console.log("Mapped predictions:", mapped);

    // Sort by confidence descending
    const sorted = [...mapped].sort((a, b) => {
      const result = b.confidence - a.confidence;
      console.log(
        `Sorting: ${b.label}(${b.confidence}) vs ${a.label}(${a.confidence}) = ${result}`,
      );
      return result;
    });

    console.log("Sorted predictions:", sorted);

    // Filter out very low confidence (but keep some threshold for display)
    const filtered = sorted.filter((pred) => pred.confidence > 0.000001); // Very low threshold
    console.log("Filtered predictions:", filtered);

    console.log("=== FINAL RESULT ===");
    console.log("Returning to ModelPredictions:", filtered);

    return filtered;
  };

  useEffect(() => {
    let interval;
    if (isRetraining) {
      interval = setInterval(async () => {
        try {
          const newMetrics = await activeLearnAPI.getMetrics();
          setMetrics({
            episodeAccuracies: newMetrics.episode_accuracies.x.map((x, i) => ({
              x,
              y: newMetrics.episode_accuracies.y[i],
            })),
            currentEpochLosses: newMetrics.current_epoch_losses.x.map(
              (x, i) => ({
                x,
                y: newMetrics.current_epoch_losses.y[i],
              }),
            ),
          });
        } catch (error) {
          console.error("Failed to fetch metrics:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isRetraining]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const [currentStatus, valStatus] = await Promise.all([
            activeLearnAPI.getStatus(),
            activeLearnAPI.getValidationStatus(), // Add this endpoint to your API service
          ]);
          setStatus(currentStatus);
          setValidationStatus(valStatus);
        } catch (error) {
          console.error("Failed to fetch status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const currentStatus = await activeLearnAPI.getStatus();
          setStatus(currentStatus);
        } catch (error) {
          console.error("Failed to fetch status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(
        async () => {
          try {
            const [status, metrics, history, lrStatus] = await Promise.all([
              activeLearnAPI.getAutomatedTrainingStatus(),
              activeLearnAPI.getMetrics(),
              activeLearnAPI.getEpisodeHistory(),
              activeLearnAPI.getLRSchedulerStatus(),
            ]);

            setAutomatedStatus(status);
            setTrainingMetrics(metrics);
            setEpisodeHistory(history.episodes);

            // Only update LR history if it exists
            if (lrStatus && lrStatus.history) {
              setLrHistory(lrStatus.history);
            }
          } catch (error) {
            console.error("Failed to fetch status:", error);
            // Don't break other functionality if LR status fails
            setLrHistory([]);
          }
        },
        status?.is_training ? 1000 : 5000,
      );
    }
    return () => clearInterval(interval);
  }, [isInitialized, status?.is_training]);

  useEffect(() => {
    let interval;
    if (isRetraining) {
      interval = setInterval(async () => {
        try {
          const newMetrics = await activeLearnAPI.getMetrics();
          setMetrics(newMetrics); // Just pass the metrics directly without transformation
        } catch (error) {
          console.error("Failed to fetch metrics:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isRetraining]);

  useEffect(() => {
    if (isInitialized && currentBatch.length > 0 && !isRetraining) {
      // Only get new batch if we haven't started labeling (completed === 0)
      if (batchStats.completed === 0) {
        getNextBatch();
        setBatchStats({
          totalImages: batchSize,
          completed: 0,
          remaining: batchSize,
          accuracy:
            metrics.episodeAccuracies[metrics.episodeAccuracies.length - 1]
              ?.y || 0.85,
          timeElapsed: "00:00",
        });
      }
    }
  }, [samplingStrategy, batchSize]);

  const handleAddLabel = () => {
    setLabels([...labels, ""]);
  };

  const handleLabelChange = (index, value) => {
    const newLabels = [...labels];
    newLabels[index] = value;
    setLabels(newLabels);
  };

  const handleRemoveLabel = (index) => {
    const newLabels = labels.filter((_, i) => i !== index);
    setLabels(newLabels);
  };

  const getNextBatch = async () => {
    try {
      console.log(
        `Getting next batch with strategy: ${samplingStrategy}, size: ${batchSize}`,
      );

      // Validate batch size to avoid None or invalid values
      const validBatchSize = parseInt(batchSize) || 16;

      const batch = await activeLearnAPI.getBatch(
        samplingStrategy,
        validBatchSize,
      );

      console.log(`Received batch with ${batch.length} images`);
      setCurrentBatch(batch);

      if (batch.length > 0) {
        // Preload the first image to check for errors
        const img = new Image();
        const imageUrl = activeLearnAPI.getImageUrl(batch[0].image_id);
        console.log(`Loading first image from URL: ${imageUrl}`);

        img.onload = () => {
          console.log("First image loaded successfully");
          setCurrentImage(imageUrl);
          setSuccessMessage(null);
        };

        img.onerror = (e) => {
          console.error("Image loading error:", e);
          setErrorMessage(
            `Failed to load image (ID: ${batch[0].image_id}). Please check image paths.`,
          );
        };

        img.src = imageUrl;
        setCurrentImageIndex(0);
      } else {
        setErrorMessage("Received empty batch. No images to display.");
      }
    } catch (error) {
      console.error("getBatch error:", error);
      setErrorMessage("Failed to get next batch: " + error.message);

      // Try fallback to random sampling
      try {
        console.log("Attempting fallback to random sampling");
        const fallbackBatch = await activeLearnAPI.getBatch(
          "random",
          Math.min(parseInt(batchSize) || 32, status.unlabeled_count || 10),
        );

        console.log(
          `Received fallback batch with ${fallbackBatch.length} images`,
        );
        setCurrentBatch(fallbackBatch);

        if (fallbackBatch.length > 0) {
          const imageUrl = activeLearnAPI.getImageUrl(
            fallbackBatch[0].image_id,
          );
          setCurrentImage(imageUrl);
          setCurrentImageIndex(0);
          setInfoMessage("Using random sampling as fallback");
        } else {
          setErrorMessage(
            "Failed to get any images. Try using a smaller batch size.",
          );
        }
      } catch (fallbackError) {
        console.error("Fallback getBatch error:", fallbackError);
        setErrorMessage(
          "Failed to get batch even with fallback strategy. Try restarting the application.",
        );
      }
    }
  };

  const handleImagesLoaded = async (
    files,
    uploadType = false,
    extraParam = ",",
    detectedLabels = [],
  ) => {
    try {
      console.log("handleImagesLoaded called with:", {
        filesCount: files.length,
        uploadType,
        extraParam,
        detectedLabels,
        firstFile: files[0], // Log the actual file object
      });

      // Auto-populate labels if we have detected labels
      if (detectedLabels && detectedLabels.length > 0) {
        console.log("Auto-populating labels from CSV:", detectedLabels);
        const newLabels = detectedLabels
          .map((label) => label.trim())
          .filter((label) => label);

        if (newLabels.length > 0) {
          setLabels(newLabels);
        }
      }

      // Handle CSV-only upload without labels
      if (uploadType === "csv") {
        const csvFile = files[0];

        if (!csvFile || !(csvFile instanceof File)) {
          throw new Error("Invalid CSV file object received");
        }

        fileDataRef.current = {
          type: "csv",
          csvFile: csvFile,
          delimiter: extraParam || ",",
        };

        setLoadedImages({
          type: "csv",
          fileCount: 1,
          file: csvFile,
          delimiter: extraParam || ",",
        });

        setSuccessMessage(
          "CSV with image paths processed. Click 'Start Project' to begin.",
        );
        return;
      }

      // Handle CSV-only upload with labels
      if (uploadType === "csv-with-labels") {
        const labelColumn = extraParam;
        const csvFile = files[0];

        if (!csvFile || !(csvFile instanceof File)) {
          throw new Error("Invalid CSV file object received");
        }

        // Store the file reference directly in ref
        fileDataRef.current = {
          type: "csv-with-labels",
          csvFile: csvFile,
          labelColumn: labelColumn,
          delimiter: ",",
          detectedLabels: detectedLabels,
        };

        // Store indicator in state (NOT the actual file)
        setLoadedImages({
          type: "csv-with-labels",
          fileCount: 1, // Make sure this is set to 1, not 0
          labelColumn: labelColumn,
          delimiter: ",",
          detectedLabels: detectedLabels,
        });

        setSuccessMessage(
          "CSV with labels processed. Click 'Start Project' to begin.",
        );
        return;
      }

      // Store the loaded files first, before any API calls
      setLoadedImages(files);
      setInfoMessage(null);

      if (detectedLabels && detectedLabels.length > 0) {
        console.log("Auto-populating labels from CSV:", detectedLabels);
        const newLabels = detectedLabels
          .map((label) => label.trim())
          .filter((label) => label);

        if (newLabels.length > 0) {
          setLabels(newLabels);
        }
      }

      // Handle other upload types...
      if (uploadType === "combined-with-labels") {
        // Store files in ref for combined uploads too
        fileDataRef.current = {
          type: "combined-with-labels",
          files: files,
          labelColumn: extraParam,
          detectedLabels: detectedLabels,
        };

        setLoadedImages({
          type: "combined-with-labels",
          fileCount: files.length,
          labelColumn: extraParam,
          detectedLabels: detectedLabels,
        });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setErrorMessage("Failed to upload: " + error.message);
    }
  };

  const handleNextImage = () => {
    if (currentImageIndex < currentBatch.length - 1) {
      setCurrentImageIndex((prev) => prev + 1);
      setCurrentImage(
        activeLearnAPI.getImageUrl(
          currentBatch[currentImageIndex + 1].image_id,
        ),
      );
    }
  };

  const handlePreviousImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex((prev) => prev - 1);
      setCurrentImage(
        activeLearnAPI.getImageUrl(
          currentBatch[currentImageIndex - 1].image_id,
        ),
      );
    }
  };

  const handleSubmitLabel = async () => {
    try {
      const imageId = currentBatch[currentImageIndex].image_id;
      const newCompleted = batchStats.completed + 1;
      const batch_complete = newCompleted === currentBatch.length;

      const result = await activeLearnAPI.submitLabel(
        imageId,
        parseInt(selectedLabel),
      );

      setBatchStats((prev) => ({
        ...prev,
        completed: newCompleted,
        remaining: prev.totalImages - newCompleted,
      }));

      setSelectedLabel("");

      if (batch_complete) {
        console.log("Starting episode training...");
        setIsRetraining(true);
        setSuccessMessage("Batch complete - Starting episode training...");

        try {
          const episodeResult = await activeLearnAPI.startEpisode(
            epochs,
            batchSize,
          );

          if (episodeResult.final_val_acc) {
            setValidationAccuracy(episodeResult.final_val_acc);
          }

          if (episodeResult.evaluation_data) {
            console.log("Evaluation data received, showing evaluation screen");
            await handleEpisodeComplete(episodeResult);
          } else {
            // Fallback to old behavior
            console.log("No evaluation data, proceeding with next batch");
            await handleStartNewBatch();
            setSuccessMessage("Episode training complete - New batch loaded");
            setIsRetraining(false);
          }
        } catch (error) {
          console.error("Episode training error:", error);
          setErrorMessage("Episode training error: " + error.message);
          setIsRetraining(false);
        }
      } else {
        handleNextImage();
      }
    } catch (error) {
      setErrorMessage("Failed to submit label: " + error.message);
    }
  };

  const handleStartNewBatch = async () => {
    try {
      setInfoMessage("Getting next batch...");
      const batchResult = await getNextBatch();

      setBatchStats({
        totalImages: batchSize,
        completed: 0,
        remaining: batchSize,
        accuracy: validationAccuracy || batchResult.accuracy || 0,
        timeElapsed: "00:00",
      });

      setIsRetraining(false);
      setSelectedLabel("");
      setIsBatchInProgress(false);
      setCurrentImageIndex(0);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage("Failed to start new batch: " + error.message);
    }
  };

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(
        async () => {
          try {
            const [status, metrics, history] = await Promise.all([
              activeLearnAPI.getAutomatedTrainingStatus(),
              activeLearnAPI.getMetrics(),
              activeLearnAPI.getEpisodeHistory(),
            ]);

            setAutomatedStatus(status);
            setTrainingMetrics(metrics);
            setEpisodeHistory(history.episodes);
            setValidationAccuracy(metrics.best_val_acc || 0);

            // Handle training completion
            if (isRetraining && !status.is_training) {
              setIsRetraining(false);
              if (status.new_batch_available) {
                await handleStartNewBatch();
              }
            }
          } catch (error) {
            console.error("Failed to fetch status:", error);
          }
        },
        status?.is_training ? 1000 : 5000,
      );
    }
    return () => clearInterval(interval);
  }, [isInitialized, isRetraining]);

  // Complete handleStartProject function for annotation.jsx
  const handleStartProject = async () => {
    if (!projectName || labels.length === 0) {
      setErrorMessage("Please set project name and labels");
      return;
    }

    // Check if we have files (handle both array and object formats)
    const hasFiles =
      (Array.isArray(loadedImages) && loadedImages.length > 0) ||
      (loadedImages &&
        typeof loadedImages === "object" &&
        loadedImages.fileCount > 0) ||
      (fileDataRef.current && fileDataRef.current.csvFile);

    if (!hasFiles) {
      setErrorMessage("Please upload images or a CSV file with image paths");
      return;
    }

    try {
      setIsLoading(true);
      setErrorMessage(null);

      // Initialize project if not already done (for imported projects, this is already done)
      if (!isInitialized) {
        console.log("Initializing new project...");
        const initResult = await activeLearnAPI.initializeProject({
          project_name: projectName,
          model_type: selectedModel,
          num_classes: labels.length,
          val_split: valSplit,
          initial_labeled_ratio: initialLabeledRatio,
          sampling_strategy: samplingStrategy,
          batch_size: parseInt(batchSize),
          epochs: parseInt(epochs),
          learning_rate: lrConfig.initial_lr,
        });
        setIsInitialized(true);
      } else {
        console.log("Using imported project model...");
      }

      // Handle CSV with labels using the ref
      if (fileDataRef.current?.type === "csv-with-labels") {
        setInfoMessage(
          `Processing CSV with annotations (using column: ${fileDataRef.current.labelColumn})...`,
        );

        const csvFile = fileDataRef.current.csvFile;

        if (!csvFile || !(csvFile instanceof File)) {
          throw new Error("CSV file object is not valid");
        }

        const expectedLabelMapping = {};
        const sourceLabels =
          fileDataRef.current.detectedLabels?.length > 0
            ? [...fileDataRef.current.detectedLabels].sort()
            : [...labels].sort();
        sourceLabels.forEach((label, index) => {
          expectedLabelMapping[label] = index;
        });
        if (sourceLabels.length > labels.length) {
          setLabels(sourceLabels);
        }

        const result = await activeLearnAPI.uploadCSVPathsWithLabels(
          csvFile,
          fileDataRef.current.labelColumn,
          fileDataRef.current.delimiter || ",",
          valSplit,
          initialLabeledRatio,
          expectedLabelMapping,
        );

        if (
          result.label_mapping &&
          Object.keys(result.label_mapping).length > 0
        ) {
          const serverLabels = Object.entries(result.label_mapping)
            .sort(([, a], [, b]) => a - b)
            .map(([name]) => name);
          setLabels(serverLabels);
        }

        setSuccessMessage(
          `Successfully loaded ${result.stats.total} images (${result.stats.labeled} with labels, ${result.stats.unlabeled} unlabeled, ${result.stats.validation} for validation)`,
        );

        // Start training if we have enough labeled data AND initial training wasn't done automatically
        if (result.stats.labeled > 0) {
          setInfoMessage("Starting initial training with annotated data...");
          try {
            const episodeResult = await activeLearnAPI.startEpisode(
              epochs,
              batchSize,
            );
            if (episodeResult.final_val_acc || episodeResult.best_accuracy) {
              const accuracy =
                episodeResult.final_val_acc || episodeResult.best_accuracy;
              setValidationAccuracy(accuracy);
              setSuccessMessage(
                `Initial training complete. Validation accuracy: ${accuracy.toFixed(
                  2,
                )}%`,
              );
            } else {
              setSuccessMessage("Initial training complete");
            }
          } catch (error) {
            console.error("Initial training error:", error);
            // If initial training fails, still proceed but show warning
            setInfoMessage(
              "Initial training encountered issues, but data is loaded. You can continue with manual training.",
            );
          }
        }
      }
      // Handle combined upload with labels
      else if (loadedImages.type === "combined-with-labels") {
        setInfoMessage(
          `Processing CSV with labels (${loadedImages.labelColumn}) and images together...`,
        );
        const result = await activeLearnAPI.uploadCombinedWithLabels(
          fileDataRef.current.files,
          fileDataRef.current.labelColumn,
          valSplit,
          initialLabeledRatio,
        );

        // Update labels from label mapping if needed
        if (result.label_mapping) {
          const newLabels = [...labels];
          Object.keys(result.label_mapping).forEach((labelText) => {
            const labelIndex = result.label_mapping[labelText];
            if (
              typeof newLabels[labelIndex] === "undefined" ||
              newLabels[labelIndex] === ""
            ) {
              while (newLabels.length <= labelIndex) {
                newLabels.push("");
              }
              newLabels[labelIndex] = labelText;
            }
          });

          if (newLabels.length > 0) {
            setLabels(newLabels);
          }
        }

        setSuccessMessage(
          `Successfully loaded ${result.stats.total} images (${result.stats.labeled} with labels, ${result.stats.unlabeled} unlabeled, ${result.stats.validation} for validation)`,
        );

        // Start training if we have enough labeled data
        if (result.stats.labeled > 0) {
          setInfoMessage("Starting initial training with annotated data...");
          try {
            const episodeResult = await activeLearnAPI.startEpisode(
              epochs,
              batchSize,
            );
            if (episodeResult.final_val_acc) {
              setValidationAccuracy(episodeResult.final_val_acc);
            }
            setSuccessMessage("Initial training complete");
          } catch (error) {
            console.error("Initial training error:", error);
            setErrorMessage("Initial training error: " + error.message);
          }
        }
      }
      // Handle combined upload without labels
      else if (loadedImages.type === "combined") {
        setInfoMessage("Processing CSV and image files together...");
        const result = await activeLearnAPI.uploadCombined(
          loadedImages.files,
          valSplit,
          initialLabeledRatio,
        );
        setSuccessMessage(
          `Successfully processed ${result.split_info.total_images} images`,
        );
      }
      // Handle CSV-only upload
      else if (loadedImages.type === "csv") {
        setInfoMessage("Processing CSV file with image paths...");

        const csvFile = loadedImages.file;
        const delimiter = loadedImages.delimiter || ",";

        try {
          const result = await activeLearnAPI.uploadCSVPaths(
            csvFile,
            delimiter,
            valSplit,
            0,
          );
          setSuccessMessage(
            `Successfully loaded ${result.split_info.total_images} images from CSV`,
          );
        } catch (error) {
          console.error("CSV upload error:", error);
          setErrorMessage(
            `Error processing CSV: ${error.message}. Try uploading images directly or using the "Upload CSV + Images Together" option.`,
          );
          return;
        }
      } else if (Array.isArray(loadedImages) && loadedImages.length > 0) {
        setInfoMessage("Setting up initial dataset...");
        const result = await activeLearnAPI.uploadData(
          loadedImages,
          valSplit,
          initialLabeledRatio,
        );
        setSuccessMessage(
          `Successfully processed ${
            result.split_info?.total_images || loadedImages.length
          } images`,
        );
      }

      // **NEW: Automatically get first batch after processing images**
      setSuccessMessage(
        "Processing complete. Getting first batch for active learning...",
      );

      try {
        await getNextBatch();
        setSuccessMessage(
          `Ready for active learning! ${
            validationAccuracy
              ? `Model accuracy: ${validationAccuracy.toFixed(2)}%. `
              : ""
          }Start labeling images.`,
        );
      } catch (batchError) {
        console.error("Error getting first batch:", batchError);
        setSuccessMessage(
          "Images processed successfully, but couldn't get first batch. Try clicking 'Start New Batch' in the status section.",
        );
      }

      setIsProjectFullyInitialized(true);
    } catch (error) {
      console.error("Project initialization error:", error);

      // Handle specific error types
      if (
        error.message &&
        error.message.includes(
          "Could not find or process any images from the CSV",
        )
      ) {
        setErrorMessage(
          `Image file path issue: The system couldn't find the image files referenced in your CSV. 
This usually happens when the paths in the CSV don't match where images are stored on your system.

Possible solutions:
1. Use absolute paths in your CSV
2. Place images in the same folder as your application
3. Update your CSV with correct relative paths
4. Try using just filenames (without directory paths) in your CSV`,
        );
      } else if (error.message && error.message.includes("422")) {
        setErrorMessage(
          "Project initialization failed. Try reloading the page and starting fresh.",
        );
      } else {
        setErrorMessage("Error: " + error.message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleEpisodeComplete = async (episodeResult) => {
    try {
      setSuccessMessage(
        "Episode training complete! Loading model evaluation...",
      );

      // Get evaluation data
      const evaluation = await activeLearnAPI.getEvaluationBatch(10);

      if (evaluation) {
        setEvaluationData(evaluation);
        setShowEvaluationScreen(true);
        setErrorMessage(null);
      } else {
        // Fallback to normal batch continuation
        await handleStartNewBatch();
        setSuccessMessage("Episode training complete - New batch loaded");
      }

      setIsRetraining(false);
    } catch (error) {
      console.error("Error getting evaluation data:", error);
      // Fallback to normal behavior
      await handleStartNewBatch();
      setSuccessMessage("Episode training complete - New batch loaded");
      setIsRetraining(false);
    }
  };

  const handleContinueFromEvaluation = async () => {
    try {
      setIsContinuing(true);
      setInfoMessage("Continuing — preparing next batch...");

      const result = await activeLearnAPI.continueFromEvaluation();

      if (result.status === "complete") {
        setSuccessMessage(result.message);
      } else {
        // Use the batch returned by the backend
        if (result.batch) {
          setCurrentBatch(result.batch);
          setCurrentImageIndex(0);
          if (result.batch.length > 0) {
            setCurrentImage(
              activeLearnAPI.getImageUrl(result.batch[0].image_id),
            );
          }
        }

        setSuccessMessage("Ready to continue active learning!");
        setBatchStats({
          totalImages: result.batch_size || batchSize,
          completed: 0,
          remaining: result.batch_size || batchSize,
          accuracy: validationAccuracy || 0,
          timeElapsed: "00:00",
        });
      }

      setShowEvaluationScreen(false);
      setEvaluationData(null);
      setIsContinuing(false);
    } catch (error) {
      console.error("Error continuing from evaluation:", error);
      setErrorMessage("Error preparing next batch: " + error.message);
      setIsContinuing(false);
    }
  };

  const handleExportFromEvaluation = async () => {
    try {
      // Update labels first
      if (labels.length > 0) {
        await activeLearnAPI.updateProjectLabels(labels);
      }

      const response = await fetch(`http://localhost:8000/export-project`, {
        method: "GET",
      });

      if (!response.ok) {
        throw new Error(`Failed to export project: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;

      const contentDisposition = response.headers.get("content-disposition");
      let filename = `${projectName}_project_${new Date()
        .toISOString()
        .slice(0, 10)}.zip`;

      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="([^"]+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setErrorMessage("Project exported successfully from evaluation!");
    } catch (error) {
      setErrorMessage("Failed to export project: " + error.message);
    }
  };

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const status = await activeLearnAPI.getStatus();
          setStatus(status);
        } catch (error) {
          console.error("Failed to fetch status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const status = await activeLearnAPI.getAutomatedTrainingStatus();
          setAutomatedStatus(status);
        } catch (error) {
          console.error("Failed to fetch automated training status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  const handleStartAutomatedTraining = async () => {
    try {
      // Check if project is initialized first
      if (!isInitialized) {
        setErrorMessage(
          "Please initialize the project with Start Project before starting automated training.",
        );
        return;
      }

      // Check if images are loaded
      if (currentBatch.length === 0) {
        setErrorMessage(
          "No images loaded. Please ensure images are loaded before starting training.",
        );
        try {
          await getNextBatch();
        } catch (error) {
          console.error("Failed to get batch before training:", error);
          return;
        }
      }

      // Make sure these values exist and are the correct type
      console.log("Starting automated training with:", {
        epochs,
        batchSize,
        samplingStrategy,
      });

      if (!epochs || !batchSize || !samplingStrategy) {
        setErrorMessage("Please set all training parameters");
        return;
      }

      // Ensure values are numbers
      const config = {
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        sampling_strategy: samplingStrategy,
        lr_config: lrConfig, // Add LR config
      };

      console.log("Sending config to API:", config);
      const response = await activeLearnAPI.startAutomatedTraining(config);
      console.log("API response:", response);

      if (response.status === "success") {
        setSuccessMessage("Automated training started successfully");
        setIsRetraining(true);
      } else {
        setErrorMessage("Failed to start training: " + response.message);
      }
    } catch (error) {
      console.error("Start training error:", error);
      setErrorMessage("Failed to start automated training: " + error.message);
    }
  };

  const handleStopAutomatedTraining = async () => {
    try {
      await activeLearnAPI.stopAutomatedTraining();
      setInfoMessage("Automated training stopped");
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  const handleSaveCheckpoint = async () => {
    try {
      setInfoMessage("Saving checkpoint...");
      const result = await activeLearnAPI.saveCheckpoint();
      setSuccessMessage("Checkpoint saved successfully!");
      // Refresh checkpoint list
      const updatedCheckpoints = await activeLearnAPI.listCheckpoints();
      setCheckpoints(updatedCheckpoints.checkpoints);
    } catch (error) {
      setErrorMessage("Failed to save checkpoint: " + error.message);
    }
  };

  const handleLoadCheckpoint = async (checkpointId) => {
    try {
      setInfoMessage("Loading checkpoint...");

      console.log("Loading checkpoint:", checkpointId);

      const response = await fetch(`http://localhost:8000/load-checkpoint`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          checkpoint_path: checkpointId,
        }),
      });

      console.log("Checkpoint load response status:", response.status);

      if (!response.ok) {
        const errorData = await response.text();
        console.error("Checkpoint load error:", errorData);
        throw new Error(
          `Failed to load checkpoint: ${response.status} - ${errorData}`,
        );
      }

      const result = await response.json();
      console.log("Checkpoint load result:", result);

      // Refresh current state
      const [status, metrics, history] = await Promise.all([
        activeLearnAPI.getStatus(),
        activeLearnAPI.getMetrics(),
        activeLearnAPI.getEpisodeHistory(),
      ]);

      setStatus(status);
      setMetrics(metrics);
      setEpisodeHistory(history.episodes);
      setSuccessMessage(
        `Checkpoint loaded successfully! Episode ${
          result.episode
        }, Accuracy: ${result.best_val_acc.toFixed(2)}%`,
      );
    } catch (error) {
      console.error("Load checkpoint error:", error);
      setErrorMessage("Failed to load checkpoint: " + error.message);
    }
  };

  useEffect(() => {
    if (isProjectFullyInitialized) {
    }
  }, [isProjectFullyInitialized]);

  useEffect(() => {
    // When currentBatch changes (like after loading a new batch)
    if (currentBatch.length > 0) {
      console.log(`New batch loaded with ${currentBatch.length} images`);

      // Reset batch stats whenever a new batch is loaded
      setBatchStats({
        totalImages: currentBatch.length,
        completed: 0,
        remaining: currentBatch.length,
        accuracy: validationAccuracy || 0.85,
        timeElapsed: "00:00",
      });

      // Reset batch progress state
      setIsBatchInProgress(true);
      setCurrentImageIndex(0);
      const loadCheckpoints = async () => {
        try {
          const result = await activeLearnAPI.listCheckpoints();
          setCheckpoints(result.checkpoints);
        } catch (error) {
          console.error("Failed to load checkpoints:", error);
        }
      };
      loadCheckpoints();
    }
  }, [currentBatch]);

  useEffect(() => {
    if (!currentBatch.length) {
      console.log("[Label Hint] No batch loaded yet");
      return;
    }
    const imageId = currentBatch[currentImageIndex]?.image_id;
    console.log(
      `[Label Hint] image_id=${imageId}, index=${currentImageIndex}, batch length=${currentBatch.length}`,
    );
    if (imageId == null) {
      console.log("[Label Hint] image_id is null/undefined");
      return;
    }
    if (typeof activeLearnAPI.getLabelHint !== "function") {
      console.error(
        "[Label Hint] getLabelHint is not defined on activeLearnAPI — check activelearning.js",
      );
      return;
    }
    activeLearnAPI
      .getLabelHint(imageId)
      .then((hint) => {
        console.log(`[Label Hint] Raw response for ${imageId}:`, hint);
        if (hint?.available) {
          const imageName = hint.file_name || `image_${imageId}`;
          console.log(
            `[Label Hint] ${imageName} → ${hint.label_name} (index ${hint.label_idx})`,
          );
        } else {
          console.log(
            `[Label Hint] available=false for image_id ${imageId} — not in ground_truth_labels on backend`,
          );
        }
      })
      .catch((err) => {
        console.error(
          `[Label Hint] Fetch failed for image_id ${imageId}:`,
          err,
        );
      });
  }, [currentBatch, currentImageIndex]);

  const hasLoadedFiles = () => {
    // Check if we have files in any format
    if (Array.isArray(loadedImages) && loadedImages.length > 0) {
      return true; // Regular file upload
    }

    if (loadedImages && typeof loadedImages === "object") {
      // CSV uploads or other special formats
      if (
        loadedImages.type === "csv-with-labels" &&
        loadedImages.fileCount > 0
      ) {
        return true;
      }
      if (
        loadedImages.type === "combined-with-labels" &&
        loadedImages.fileCount > 0
      ) {
        return true;
      }
      if (loadedImages.type === "combined" && loadedImages.fileCount > 0) {
        return true;
      }
      if (loadedImages.type === "csv" && loadedImages.fileCount > 0) {
        return true;
      }
    }

    // Check if we have file data in the ref (CSV uploads)
    if (fileDataRef.current && fileDataRef.current.csvFile) {
      return true;
    }

    return false;
  };

  // Add this function to auto-start batching
  const handleAutoStartBatch = async () => {
    if (isInitialized && !isRetraining && currentBatch.length === 0) {
      try {
        setInfoMessage("Automatically loading next batch...");
        await getNextBatch();
        setSuccessMessage(
          `New batch loaded! Ready to continue active learning.`,
        );
      } catch (error) {
        console.error("Auto-batch loading failed:", error);
        setErrorMessage(
          "Couldn't automatically load batch. Try manual batch loading.",
        );
      }
    }
  };

  // Use this effect to auto-start batching when conditions are right
  useEffect(() => {
    if (
      isProjectFullyInitialized &&
      currentBatch.length === 0 &&
      !isRetraining
    ) {
      const timer = setTimeout(() => {
        handleAutoStartBatch();
      }, 1000); // Small delay to ensure everything is ready

      return () => clearTimeout(timer);
    }
  }, [isProjectFullyInitialized, currentBatch.length, isRetraining]);


  const [isCleaningUp, setIsCleaningUp] = useState(false); // next to isContinuing

  const handleCleanupTempFiles = async () => {
    const confirmed = window.confirm(
      "This deletes saved checkpoints and temp files for this project on the " +
        "server (output/" + projectName + "/). Your current in-memory model and " +
        "labels are not affected, and training can continue right after. Continue?"
    );
    if (!confirmed) return;

    setIsCleaningUp(true);
    try {
      const result = await activeLearnAPI.cleanupProject();
      setSuccessMessage(
        `Temp files cleared (${result.removed.length ? result.removed.join(", ") : "nothing to remove"}).`
      );
    } catch (error) {
      setErrorMessage("Failed to clean up temp files: " + error.message);
    } finally {
      setIsCleaningUp(false);
    }
  };


  return (
    <>
      {showEvaluationScreen ? (
        <ModelEvaluationScreen
          evaluationData={evaluationData}
          onContinueTraining={handleContinueFromEvaluation}
          isContinuing={isContinuing}
          onExportModel={handleExportFromEvaluation}
          labels={labels}
          currentEpisode={status.current_episode || 0}
        />
      ) : (
        <>
          <div className="flex px-5 pt-5 pb-3 h-1/6 items-center">
            <div className="text-xs md:text-sm text-gray-500">
              <span className="font-medium text-gray-500">
                Developed at Harvard Ophthalmology AI Lab, 2026
              </span>
              <br />
              By Srikar Kusumanchi & Mohammad Eslami
            </div>
          </div>

          <div className="container mx-auto p-6">
            <div className="grid grid-cols-2 gap-6">
              {/* Left Column - Project Configuration */}
              <div>
                <Tabs
                  value={activeTab}
                  onValueChange={setActiveTab}
                  className="mb-6"
                >
                  <TabsList className="grid grid-cols-2">
                    <TabsTrigger value="new">New Project</TabsTrigger>
                    <TabsTrigger value="import">Import Project</TabsTrigger>
                  </TabsList>
                  <TabsContent value="new">
                    <Card>
                      <CardHeader>
                        <CardTitle>Project Configuration</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-6">
                          <div className="space-y-3">
                            <RadioGroup
                              value={projectType}
                              onValueChange={setProjectType}
                              className="flex gap-6"
                            >
                              <div className="flex items-center space-x-2">
                                <RadioGroupItem value="new" id="new-model" />
                                <Label htmlFor="new-model">
                                  Create New Model
                                </Label>
                              </div>
                              <div className="flex items-center space-x-2">
                                <RadioGroupItem
                                  value="pretrained"
                                  id="pretrained-model"
                                />
                                <Label htmlFor="pretrained-model">
                                  Import Pretrained Model
                                </Label>
                              </div>
                            </RadioGroup>
                          </div>
                          {projectType === "new" ? (
                            <>
                              {/* New Model Configuration */}
                              <div>
                                <Label htmlFor="project-name">
                                  Project Name
                                </Label>
                                <Input
                                  id="project-name"
                                  value={projectName}
                                  onChange={(e) =>
                                    setProjectName(e.target.value)
                                  }
                                  className="mt-1"
                                  disabled={isInitialized}
                                  placeholder="Enter project name"
                                />
                              </div>

                              <div>
                                <Label>Model Selection</Label>
                                <Select
                                  value={selectedModel}
                                  onValueChange={setSelectedModel}
                                  disabled={isInitialized}
                                >
                                  <SelectTrigger>
                                    <SelectValue placeholder="Select model" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="resnet18">
                                      ResNet18
                                    </SelectItem>
                                    <SelectItem value="resnet50">
                                      ResNet50
                                    </SelectItem>
                                    <SelectItem value="dinov2">
                                      DINOv2 (ViT-B/14)
                                    </SelectItem>
                                    <SelectItem value="custom">
                                      Custom Model
                                    </SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>

                              {/* Label Management */}
                              <div>
                                <Label>Labels</Label>
                                {labels.map((label, index) => (
                                  <div key={index} className="flex gap-2 mt-2">
                                    <Input
                                      value={label}
                                      onChange={(e) =>
                                        handleLabelChange(index, e.target.value)
                                      }
                                      placeholder={`Label ${index + 1}`}
                                    />
                                    {labels.length > 1 && (
                                      <Button
                                        variant="outline"
                                        size="icon"
                                        onClick={() => handleRemoveLabel(index)}
                                      >
                                        <Trash2 className="h-4 w-4" />
                                      </Button>
                                    )}
                                  </div>
                                ))}
                                <Button
                                  variant="outline"
                                  onClick={handleAddLabel}
                                  className="w-full mt-2"
                                >
                                  <Plus className="h-4 w-4 mr-2" />
                                  Add Label
                                </Button>
                              </div>

                              {/* Sampling Strategy Controls */}
                              {/* Training Parameters Card */}
                              <Card className="p-4">
                                <CardHeader className="pb-2 px-0">
                                  <CardTitle className="text-lg">
                                    Training Parameters
                                  </CardTitle>
                                </CardHeader>
                                <CardContent className="px-0">
                                  <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                      <Label>Sampling Strategy</Label>
                                      <Select
                                        value={samplingStrategy}
                                        onValueChange={setSamplingStrategy}
                                        disabled={
                                          isInitialized ||
                                          batchStats.completed > 0
                                        }
                                      >
                                        <SelectTrigger>
                                          <SelectValue placeholder="Select strategy" />
                                        </SelectTrigger>
                                        <SelectContent>
                                          <SelectItem value="least_confidence">
                                            Least Confidence
                                          </SelectItem>
                                          <SelectItem value="margin">
                                            Margin Sampling
                                          </SelectItem>
                                          <SelectItem value="entropy">
                                            Entropy
                                          </SelectItem>
                                          <SelectItem value="diversity">
                                            Diversity-based
                                          </SelectItem>
                                        </SelectContent>
                                      </Select>
                                    </div>

                                    <div className="space-y-2">
                                      <Label>Learning Rate Strategy</Label>
                                      <Select
                                        value={lrConfig.strategy}
                                        onValueChange={(value) =>
                                          setLrConfig((prev) => ({
                                            ...prev,
                                            strategy: value,
                                          }))
                                        }
                                        disabled={
                                          isInitialized ||
                                          batchStats.completed > 0
                                        }
                                      >
                                        <SelectTrigger>
                                          <SelectValue placeholder="Select LR strategy" />
                                        </SelectTrigger>
                                        <SelectContent>
                                          <SelectItem value="plateau">
                                            Reduce on Plateau
                                          </SelectItem>
                                          <SelectItem value="cosine">
                                            Cosine Annealing
                                          </SelectItem>
                                          <SelectItem value="warmup">
                                            One Cycle with Warmup
                                          </SelectItem>
                                          <SelectItem value="step">
                                            Step Decay
                                          </SelectItem>
                                        </SelectContent>
                                      </Select>
                                    </div>
                                  </div>

                                  <div className="grid grid-cols-3 gap-4 mt-4">
                                    <div className="space-y-2">
                                      <Label>Initial Learning Rate</Label>
                                      <Input
                                        type="number"
                                        value={lrConfig.initial_lr}
                                        onChange={(e) =>
                                          setLrConfig((prev) => ({
                                            ...prev,
                                            initial_lr: parseFloat(
                                              e.target.value,
                                            ),
                                          }))
                                        }
                                        min={0.0001}
                                        max={0.1}
                                        step={0.0001}
                                        disabled={isInitialized || isRetraining}
                                      />
                                    </div>

                                    <div className="space-y-2">
                                      <Label>Batch Size</Label>
                                      <Input
                                        type="number"
                                        value={batchSize}
                                        onChange={(e) => {
                                          const newBatchSize = Number(
                                            e.target.value,
                                          );
                                          setBatchSize(newBatchSize);
                                          setBatchStats((prev) => ({
                                            ...prev,
                                            totalImages: newBatchSize,
                                            remaining: newBatchSize,
                                          }));
                                        }}
                                        min={1}
                                        max={100}
                                        disabled={
                                          isInitialized ||
                                          isRetraining ||
                                          batchStats.completed > 0
                                        }
                                      />
                                    </div>

                                    <div className="space-y-2">
                                      <Label>Epochs</Label>
                                      <Input
                                        type="number"
                                        value={epochs}
                                        onChange={(e) => {
                                          const newEpochs = Number(
                                            e.target.value,
                                          );
                                          setEpochs(newEpochs);
                                        }}
                                        min={1}
                                        disabled={isInitialized || isRetraining}
                                      />
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>

                              {/* Data Split Configuration */}
                              <div className="space-y-2">
                                <Label>Validation Split</Label>
                                <div className="flex items-center space-x-2">
                                  <Input
                                    type="range"
                                    min="0.0"
                                    max="0.3"
                                    step="0.01"
                                    value={valSplit}
                                    onChange={(e) =>
                                      setValSplit(parseFloat(e.target.value))
                                    }
                                    disabled={isInitialized}
                                    className="flex-1"
                                  />
                                  <span className="text-sm w-16 text-right">
                                    {(valSplit * 100).toFixed(0)}%
                                  </span>
                                </div>
                                {valSplit < 0.05 && (
                                  <p className="text-xs text-amber-600 flex items-center gap-1">
                                    ⚠ Validation set is very small (&lt;5%).
                                    Accuracy metrics may be unreliable.
                                  </p>
                                )}
                              </div>

                              {/* Data Split Preview */}
                              {loadedImages.length > 0 && (
                                <Card className="bg-secondary/50">
                                  <CardContent className="pt-4">
                                    <h4 className="font-medium mb-2">
                                      Data Split Preview
                                    </h4>
                                    <div className="space-y-1 text-sm">
                                      <div className="flex justify-between">
                                        <span>Total Images:</span>
                                        <span>{loadedImages.length}</span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span>Validation Set:</span>
                                        <span>
                                          {Math.round(
                                            loadedImages.length * valSplit,
                                          )}{" "}
                                          images
                                        </span>
                                      </div>
                                      <div className="flex justify-between">
                                        <span>Unlabeled Pool:</span>
                                        <span>
                                          {loadedImages.length -
                                            Math.round(
                                              loadedImages.length * valSplit,
                                            )}{" "}
                                          images
                                        </span>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              )}

                              <ImageLoader
                                onImagesLoaded={handleImagesLoaded}
                                onError={setErrorMessage}
                              />
                            </>
                          ) : (
                            <>
                              <PretrainedModelImport
                                onImportSuccess={(result) => {
                                  console.log("=== IMPORT DEBUG ===");
                                  console.log("Full result:", result);

                                  setProjectName(
                                    result.project_info.project_name,
                                  );
                                  const importedModelType =
                                    result.project_info.model_type ||
                                    "resnet50";
                                  setSelectedModel(importedModelType);
                                  setIsInitialized(true);

                                  // Restore labels from imported project
                                  if (
                                    result.labels &&
                                    result.labels.label_names
                                  ) {
                                    setLabels(result.labels.label_names);
                                    console.log(
                                      "Restored labels:",
                                      result.labels.label_names,
                                    );
                                  } else {
                                    const numClasses =
                                      result.project_info.num_classes || 2;
                                    const defaultLabels = Array.from(
                                      { length: numClasses },
                                      (_, i) => `Class ${i + 1}`,
                                    );
                                    setLabels(defaultLabels);
                                  }

                                  // Update hyperparameters
                                  if (result.hyperparameters) {
                                    setSamplingStrategy(
                                      result.hyperparameters
                                        .sampling_strategy ||
                                        "least_confidence",
                                    );
                                    setBatchSize(
                                      result.hyperparameters.batch_size || 32,
                                    );
                                    setEpochs(
                                      result.hyperparameters.epochs || 10,
                                    );
                                    setValSplit(
                                      result.hyperparameters.validation_split ||
                                        0.2,
                                    );
                                    setInitialLabeledRatio(
                                      result.hyperparameters
                                        .initial_labeled_ratio || 0.1,
                                    );
                                  }

                                  // Update metrics
                                  if (result.project_info) {
                                    setValidationAccuracy(
                                      result.project_info
                                        .best_validation_accuracy || 0,
                                    );
                                  }

                                  // **NEW: Handle automatic image loading**
                                  if (
                                    result.images_loaded &&
                                    result.project_ready
                                  ) {
                                    setIsProjectFullyInitialized(true);
                                    setSuccessMessage(
                                      `Project imported successfully! Model: ${importedModelType}. 
       Loaded ${result.dataset_stats.loaded_from_annotations} images automatically. 
       Getting first batch for active learning...`,
                                    );

                                    // Auto-start by getting the first batch
                                    setTimeout(async () => {
                                      try {
                                        await getNextBatch();
                                        setSuccessMessage(
                                          `Ready for active learning! ${result.dataset_stats.current_labeled} labeled, ${result.dataset_stats.current_unlabeled} unlabeled images loaded.`,
                                        );
                                      } catch (error) {
                                        console.error(
                                          "Error getting first batch:",
                                          error,
                                        );
                                        setErrorMessage(
                                          `Images loaded but couldn't get first batch: ${error.message}. Try manually starting a new batch.`,
                                        );
                                      }
                                    }, 1000);
                                  } else if (
                                    result.dataset_stats
                                      .loaded_from_annotations > 0
                                  ) {
                                    // Some images loaded but project not ready (maybe all labeled)
                                    setErrorMessage(
                                      `Project imported with ${result.dataset_stats.loaded_from_annotations} images, but no unlabeled data for active learning. Upload more images or check your data split.`,
                                    );
                                  } else {
                                    // No images loaded automatically
                                    setErrorMessage(
                                      `${result.message} Images were not found in the expected locations. You can upload new images to continue.`,
                                    );
                                  }

                                  // Fetch current state
                                  activeLearnAPI
                                    .getStatus()
                                    .then(setStatus)
                                    .catch(console.error);
                                  activeLearnAPI
                                    .getMetrics()
                                    .then(setMetrics)
                                    .catch(console.error);
                                }}
                                onError={(errorMsg) => {
                                  setErrorMessage(errorMsg);
                                }}
                              />

                              {selectedModel === "custom" && (
                                <div className="p-3 bg-blue-50 border border-blue-200 rounded">
                                  <p className="text-sm text-blue-700">
                                    <strong>Custom Model:</strong> Upload any
                                    PyTorch model file (.pt, .pth). The system
                                    will attempt to automatically detect the
                                    architecture and adapt it for your classes.
                                  </p>
                                </div>
                              )}

                              {modelLoaded && (
                                <>
                                  <ModelAdaptationControls
                                    onAdaptSuccess={(result) => {
                                      setSuccessMessage(
                                        `Model adaptation successful using ${result.adaptation_type} strategy`,
                                      );
                                    }}
                                    onError={(errorMsg) => {
                                      setErrorMessage(
                                        `Adaptation error: ${errorMsg}`,
                                      );
                                    }}
                                    disabled={!isInitialized || isRetraining}
                                  />

                                  {/* Label Management for Pretrained Model */}
                                  <div>
                                    <Label>Labels</Label>
                                    <p className="text-sm text-gray-500 mb-2">
                                      Define your labels here
                                    </p>
                                    {labels.map((label, index) => (
                                      <div
                                        key={index}
                                        className="flex gap-2 mt-2"
                                      >
                                        <Input
                                          value={label}
                                          onChange={(e) =>
                                            handleLabelChange(
                                              index,
                                              e.target.value,
                                            )
                                          }
                                          placeholder={`Label ${index + 1}`}
                                        />
                                        {labels.length > 1 && (
                                          <Button
                                            variant="outline"
                                            size="icon"
                                            onClick={() =>
                                              handleRemoveLabel(index)
                                            }
                                          >
                                            <Trash2 className="h-4 w-4" />
                                          </Button>
                                        )}
                                      </div>
                                    ))}
                                    <div className="flex gap-2 mt-2">
                                      <Button
                                        variant="outline"
                                        onClick={handleAddLabel}
                                        className="flex-1"
                                      >
                                        <Plus className="h-4 w-4 mr-2" />
                                        Add Label
                                      </Button>
                                    </div>
                                  </div>

                                  {/* Hyperparameters for Pretrained Model */}
                                  <div className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4">
                                      <div className="space-y-2">
                                        <Label>Sampling Strategy</Label>
                                        <Select
                                          value={samplingStrategy}
                                          onValueChange={setSamplingStrategy}
                                          disabled={batchStats.completed > 0}
                                        >
                                          <SelectTrigger>
                                            <SelectValue placeholder="Select strategy" />
                                          </SelectTrigger>
                                          <SelectContent>
                                            <SelectItem value="least_confidence">
                                              Least Confidence
                                            </SelectItem>
                                            <SelectItem value="margin">
                                              Margin Sampling
                                            </SelectItem>
                                            <SelectItem value="entropy">
                                              Entropy
                                            </SelectItem>
                                            <SelectItem value="diversity">
                                              Diversity-based
                                            </SelectItem>
                                          </SelectContent>
                                        </Select>
                                      </div>

                                      <div className="space-y-2">
                                        <Label>Learning Rate Strategy</Label>
                                        <Select
                                          value={lrConfig.strategy}
                                          onValueChange={(value) =>
                                            setLrConfig((prev) => ({
                                              ...prev,
                                              strategy: value,
                                            }))
                                          }
                                          disabled={
                                            isInitialized ||
                                            batchStats.completed > 0
                                          }
                                        >
                                          <SelectTrigger>
                                            <SelectValue placeholder="Select LR strategy" />
                                          </SelectTrigger>
                                          <SelectContent>
                                            <SelectItem value="plateau">
                                              Reduce on Plateau
                                            </SelectItem>
                                            <SelectItem value="cosine">
                                              Cosine Annealing
                                            </SelectItem>
                                            <SelectItem value="warmup">
                                              One Cycle with Warmup
                                            </SelectItem>
                                            <SelectItem value="step">
                                              Step Decay
                                            </SelectItem>
                                          </SelectContent>
                                        </Select>
                                      </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-4">
                                      <div className="space-y-2">
                                        <Label>Initial Learning Rate</Label>
                                        <Input
                                          type="number"
                                          value={lrConfig.initial_lr}
                                          onChange={(e) =>
                                            setLrConfig((prev) => ({
                                              ...prev,
                                              initial_lr: parseFloat(
                                                e.target.value,
                                              ),
                                            }))
                                          }
                                          min={0.0001}
                                          max={0.1}
                                          step={0.0001}
                                          disabled={
                                            isInitialized ||
                                            batchStats.completed > 0
                                          }
                                        />
                                      </div>

                                      <div className="space-y-2">
                                        <Label>Batch Size</Label>
                                        <Input
                                          type="number"
                                          value={batchSize}
                                          onChange={(e) => {
                                            const newBatchSize = Number(
                                              e.target.value,
                                            );
                                            setBatchSize(newBatchSize);
                                            setBatchStats((prev) => ({
                                              ...prev,
                                              totalImages: newBatchSize,
                                              remaining: newBatchSize,
                                            }));
                                          }}
                                          min={1}
                                          max={100}
                                          disabled={
                                            isRetraining ||
                                            batchStats.completed > 0
                                          }
                                        />
                                      </div>

                                      <div className="space-y-2">
                                        <Label>Epochs</Label>
                                        <Input
                                          type="number"
                                          value={epochs}
                                          onChange={(e) => {
                                            const newEpochs = Number(
                                              e.target.value,
                                            );
                                            setEpochs(newEpochs);
                                          }}
                                          min={5}
                                          disabled={
                                            isInitialized ||
                                            batchStats.completed > 0
                                          }
                                        />
                                      </div>
                                    </div>

                                    {/* Data Split Configuration for Pretrained */}
                                    <div className="space-y-2">
                                      <Label>Validation Split</Label>
                                      <div className="flex items-center space-x-2">
                                        <Input
                                          type="range"
                                          min="0.0"
                                          max="0.3"
                                          step="0.01"
                                          value={valSplit}
                                          onChange={(e) =>
                                            setValSplit(
                                              parseFloat(e.target.value),
                                            )
                                          }
                                          className="flex-1"
                                        />
                                        <span className="text-sm w-16 text-right">
                                          {(valSplit * 100).toFixed(0)}%
                                        </span>
                                      </div>
                                      {valSplit < 0.05 && (
                                        <p className="text-xs text-amber-600 flex items-center gap-1">
                                          ⚠ Validation set is very small
                                          (&lt;5%). Accuracy metrics may be
                                          unreliable.
                                        </p>
                                      )}
                                    </div>
                                  </div>

                                  <ImageLoader
                                    onImagesLoaded={handleImagesLoaded}
                                    onError={setErrorMessage}
                                  />
                                </>
                              )}
                            </>
                          )}

                          {message.text &&
                            (() => {
                              const alertProps = getAlertProps(message.type);
                              const IconComponent = alertProps.icon;
                              return (
                                <Alert
                                  variant={alertProps.variant}
                                  className={alertProps.className}
                                >
                                  <IconComponent className="h-4 w-4" />
                                  <AlertDescription>
                                    {message.text}
                                  </AlertDescription>
                                </Alert>
                              );
                            })()}

                          <Button
                            className="w-full"
                            onClick={handleStartProject}
                            disabled={
                              !projectName ||
                              labels.length === 0 ||
                              !hasLoadedFiles() ||
                              isProjectFullyInitialized ||
                              isLoading
                            }
                          >
                            {isLoading
                              ? "Initializing Project..."
                              : isInitialized && !isProjectFullyInitialized
                                ? "Start Project with Pretrained Model"
                                : "Start Project"}
                          </Button>
                          {isInitialized && (
                            <AutomatedTrainingControls
                              status={automatedStatus}
                              metrics={trainingMetrics}
                              disabled={!isInitialized}
                              episode_history={episodeHistory}
                            />
                          )}

                          <div>
                            <ActiveLearningStatus
                              status={status}
                              onStartNewBatch={handleStartNewBatch}
                            />
                          </div>
                          {/* 
                          <CheckpointControls
                            onSave={handleSaveCheckpoint}
                            onLoad={handleLoadCheckpoint}
                            checkpoints={checkpoints}
                          />
*/}
                          <div className="space-y-4">
                            <div className="flex gap-4">
                              <Button
                                onClick={async () => {
                                  try {
                                    setInfoMessage(
                                      "Running predictions and packaging your model — this may take a moment. Your ZIP file will download automatically once it is ready.",
                                    );

                                    // ALWAYS send current labels before export
                                    console.log(
                                      "Sending labels to backend before export:",
                                      labels,
                                    );

                                    if (labels.length > 0) {
                                      const labelResponse = await fetch(
                                        `http://localhost:8000/update-project-labels`,
                                        {
                                          method: "POST",
                                          headers: {
                                            "Content-Type": "application/json",
                                          },
                                          body: JSON.stringify({
                                            labels: labels,
                                          }),
                                        },
                                      );

                                      if (!labelResponse.ok) {
                                        console.warn(
                                          "Failed to update labels, but continuing with export",
                                        );
                                      }
                                    }

                                    const response = await fetch(
                                      `http://localhost:8000/export-project`,
                                      {
                                        method: "GET",
                                      },
                                    );

                                    if (!response.ok) {
                                      throw new Error(
                                        `Failed to export project: ${response.statusText}`,
                                      );
                                    }

                                    const blob = await response.blob();
                                    const url =
                                      window.URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.href = url;

                                    const contentDisposition =
                                      response.headers.get(
                                        "content-disposition",
                                      );
                                    let filename = `${projectName}_project_${new Date()
                                      .toISOString()
                                      .slice(0, 10)}.zip`;

                                    if (contentDisposition) {
                                      const filenameMatch =
                                        contentDisposition.match(
                                          /filename="([^"]+)"/,
                                        );
                                      if (filenameMatch) {
                                        filename = filenameMatch[1];
                                      }
                                    }

                                    a.download = filename;
                                    document.body.appendChild(a);
                                    a.click();
                                    window.URL.revokeObjectURL(url);
                                    document.body.removeChild(a);

                                    setSuccessMessage(
                                      "Project exported successfully!",
                                    );
                                  } catch (error) {
                                    setErrorMessage(
                                      "Failed to export project: " +
                                        error.message,
                                    );
                                  }
                                }}
                                disabled={!isInitialized || !projectName}
                              >
                                <Download className="h-4 w-4 mr-2" />
                                Export Project with Latest Checkpoint and Labels
                              </Button>

                                <Button
                                  variant="outline"
                                  onClick={handleCleanupTempFiles}
                                  disabled={!isInitialized || !projectName || isCleaningUp}
                                >
                                  <Trash2 className="h-4 w-4 mr-2" />
                                  {isCleaningUp ? "Clearing..." : "Clear Temp Files"}
                                </Button>

                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="import">
                    <ProjectImport
                      onImportSuccess={(result) => {
                        console.log("=== IMPORT DEBUG ===");
                        console.log("Full result:", result);

                        setProjectName(result.project_info.project_name);
                        const importedModelType =
                          result.project_info.model_type || "resnet50";
                        setSelectedModel(importedModelType);
                        setIsInitialized(true);

                        // Restore labels from imported project
                        if (result.labels && result.labels.label_names) {
                          setLabels(result.labels.label_names);
                          console.log(
                            "Restored labels:",
                            result.labels.label_names,
                          );
                        } else {
                          const numClasses =
                            result.project_info.num_classes || 2;
                          const defaultLabels = Array.from(
                            { length: numClasses },
                            (_, i) => `Class ${i + 1}`,
                          );
                          setLabels(defaultLabels);
                        }

                        // Update hyperparameters
                        if (result.hyperparameters) {
                          setSamplingStrategy(
                            result.hyperparameters.sampling_strategy ||
                              "least_confidence",
                          );
                          setBatchSize(result.hyperparameters.batch_size || 32);
                          setEpochs(result.hyperparameters.epochs || 10);
                          setValSplit(
                            result.hyperparameters.validation_split || 0.2,
                          );
                          setInitialLabeledRatio(
                            result.hyperparameters.initial_labeled_ratio || 0.1,
                          );
                        }

                        // Update metrics
                        if (result.project_info) {
                          setValidationAccuracy(
                            result.project_info.best_validation_accuracy || 0,
                          );
                        }

                        // **NEW: Handle automatic image loading**
                        if (result.images_loaded && result.project_ready) {
                          setIsProjectFullyInitialized(true);
                          setSuccessMessage(
                            `Project imported successfully! Model: ${importedModelType}. 
       Loaded ${result.dataset_stats.loaded_from_annotations} images automatically. 
       Getting first batch for active learning...`,
                          );

                          // Auto-start by getting the first batch
                          setTimeout(async () => {
                            try {
                              await getNextBatch();
                              setSuccessMessage(
                                `Ready for active learning! ${result.dataset_stats.current_labeled} labeled, ${result.dataset_stats.current_unlabeled} unlabeled images loaded.`,
                              );
                            } catch (error) {
                              console.error(
                                "Error getting first batch:",
                                error,
                              );
                              setErrorMessage(
                                `Images loaded but couldn't get first batch: ${error.message}. Try manually starting a new batch.`,
                              );
                            }
                          }, 1000);
                        } else if (
                          result.dataset_stats.loaded_from_annotations > 0
                        ) {
                          // Some images loaded but project not ready (maybe all labeled)
                          setErrorMessage(
                            `Project imported with ${result.dataset_stats.loaded_from_annotations} images, but no unlabeled data for active learning. Upload more images or check your data split.`,
                          );
                        } else {
                          // No images loaded automatically
                          setErrorMessage(
                            `${result.message} Images were not found in the expected locations. You can upload new images to continue.`,
                          );
                        }

                        // Fetch current state
                        activeLearnAPI
                          .getStatus()
                          .then(setStatus)
                          .catch(console.error);
                        activeLearnAPI
                          .getMetrics()
                          .then(setMetrics)
                          .catch(console.error);
                      }}
                      onError={(errorMsg) => {
                        setErrorMessage(errorMsg);
                      }}
                    />

                    {/* Show full project controls after successful import */}
                    {isInitialized && (
                      <Card className="mt-6">
                        <CardHeader>
                          <CardTitle>Imported Project Configuration</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-6">
                            {/* Project Name - Read Only Display */}
                            <div>
                              <Label>Project Name</Label>
                              <Input
                                value={projectName}
                                onChange={(e) => setProjectName(e.target.value)}
                                className="mt-1"
                                placeholder="Project name"
                              />
                            </div>

                            {/* Model Info - Read Only Display */}
                            <div>
                              <Label>Model Type</Label>
                              <Input
                                value={selectedModel}
                                readOnly
                                className="mt-1 bg-gray-50"
                              />
                            </div>

                            {/* Labels Management */}
                            <div>
                              <Label>Labels</Label>
                              <p className="text-sm text-gray-600 mb-2">
                                You can modify these labels for your dataset
                              </p>
                              {labels.map((label, index) => (
                                <div key={index} className="flex gap-2 mt-2">
                                  <Input
                                    value={label}
                                    onChange={(e) =>
                                      handleLabelChange(index, e.target.value)
                                    }
                                    placeholder={`Label ${index + 1}`}
                                  />
                                  {labels.length > 1 && (
                                    <Button
                                      variant="outline"
                                      size="icon"
                                      onClick={() => handleRemoveLabel(index)}
                                    >
                                      <Trash2 className="h-4 w-4" />
                                    </Button>
                                  )}
                                </div>
                              ))}
                              <div className="flex gap-2 mt-2">
                                <Button
                                  variant="outline"
                                  onClick={handleAddLabel}
                                  className="flex-1"
                                >
                                  <Plus className="h-4 w-4 mr-2" />
                                  Add Label
                                </Button>
                              </div>
                            </div>

                            {/* Hyperparameters */}
                            <div className="space-y-4">
                              <h4 className="text-lg font-medium">
                                Training Parameters
                              </h4>

                              <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                  <Label>Sampling Strategy</Label>
                                  <Select
                                    value={samplingStrategy}
                                    onValueChange={setSamplingStrategy}
                                    disabled={batchStats.completed > 0}
                                  >
                                    <SelectTrigger>
                                      <SelectValue placeholder="Select strategy" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="least_confidence">
                                        Least Confidence
                                      </SelectItem>
                                      <SelectItem value="margin">
                                        Margin Sampling
                                      </SelectItem>
                                      <SelectItem value="entropy">
                                        Entropy
                                      </SelectItem>
                                      <SelectItem value="diversity">
                                        Diversity-based
                                      </SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>

                                <div className="space-y-2">
                                  <Label>Learning Rate Strategy</Label>
                                  <Select
                                    value={lrConfig.strategy}
                                    onValueChange={(value) =>
                                      setLrConfig((prev) => ({
                                        ...prev,
                                        strategy: value,
                                      }))
                                    }
                                    disabled={
                                      isInitialized || batchStats.completed > 0
                                    }
                                  >
                                    <SelectTrigger>
                                      <SelectValue placeholder="Select LR strategy" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="plateau">
                                        Reduce on Plateau
                                      </SelectItem>
                                      <SelectItem value="cosine">
                                        Cosine Annealing
                                      </SelectItem>
                                      <SelectItem value="warmup">
                                        One Cycle with Warmup
                                      </SelectItem>
                                      <SelectItem value="step">
                                        Step Decay
                                      </SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                              </div>

                              <div className="grid grid-cols-3 gap-4">
                                <div className="space-y-2">
                                  <Label>Initial Learning Rate</Label>
                                  <Input
                                    type="number"
                                    value={lrConfig.initial_lr}
                                    onChange={(e) =>
                                      setLrConfig((prev) => ({
                                        ...prev,
                                        initial_lr: parseFloat(e.target.value),
                                      }))
                                    }
                                    min={0.0001}
                                    max={0.1}
                                    step={0.0001}
                                    disabled={
                                      isInitialized || batchStats.completed > 0
                                    }
                                  />
                                </div>

                                <div className="space-y-2">
                                  <Label>Batch Size</Label>
                                  <Input
                                    type="number"
                                    value={batchSize}
                                    onChange={(e) => {
                                      const newBatchSize = Number(
                                        e.target.value,
                                      );
                                      setBatchSize(newBatchSize);
                                      setBatchStats((prev) => ({
                                        ...prev,
                                        totalImages: newBatchSize,
                                        remaining: newBatchSize,
                                      }));
                                    }}
                                    min={1}
                                    max={100}
                                    disabled={
                                      isRetraining || batchStats.completed > 0
                                    }
                                  />
                                </div>

                                <div className="space-y-2">
                                  <Label>Epochs</Label>
                                  <Input
                                    type="number"
                                    value={epochs}
                                    onChange={(e) => {
                                      const newEpochs = Number(e.target.value);
                                      setEpochs(newEpochs);
                                    }}
                                    min={5}
                                    disabled={
                                      isInitialized || batchStats.completed > 0
                                    }
                                  />
                                </div>
                              </div>

                              {/* Data Split Configuration */}
                              <div className="space-y-2">
                                <Label>Validation Split</Label>
                                <div className="flex items-center space-x-2">
                                  <Input
                                    type="range"
                                    min="0.0"
                                    max="0.3"
                                    step="0.01"
                                    value={valSplit}
                                    onChange={(e) =>
                                      setValSplit(parseFloat(e.target.value))
                                    }
                                    className="flex-1"
                                  />
                                  <span className="text-sm w-16 text-right">
                                    {(valSplit * 100).toFixed(0)}%
                                  </span>
                                </div>
                                {valSplit < 0.05 && (
                                  <p className="text-xs text-amber-600 flex items-center gap-1">
                                    ⚠ Validation set is very small (&lt;5%).
                                    Accuracy metrics may be unreliable.
                                  </p>
                                )}
                              </div>
                            </div>

                            {/* Image Loader */}
                            <ImageLoader
                              onImagesLoaded={handleImagesLoaded}
                              onError={setErrorMessage}
                            />

                            {message.text &&
                              (() => {
                                const alertProps = getAlertProps(message.type);
                                const IconComponent = alertProps.icon;
                                return (
                                  <Alert
                                    variant={alertProps.variant}
                                    className={alertProps.className}
                                  >
                                    <IconComponent className="h-4 w-4" />
                                    <AlertDescription>
                                      {message.text}
                                    </AlertDescription>
                                  </Alert>
                                );
                              })()}

                            {/* Start Project Button */}
                            <Button
                              className="w-full"
                              onClick={handleStartProject}
                              disabled={
                                !projectName ||
                                labels.length === 0 ||
                                !hasLoadedFiles() ||
                                isProjectFullyInitialized ||
                                isLoading
                              }
                            >
                              {isLoading
                                ? "Initializing Project..."
                                : isProjectFullyInitialized
                                  ? "Project Started"
                                  : "Start Project"}
                            </Button>

                            {/* Training Controls */}
                            {isProjectFullyInitialized && (
                              <>
                                <AutomatedTrainingControls
                                  status={automatedStatus}
                                  metrics={trainingMetrics}
                                  disabled={!isInitialized}
                                  episode_history={episodeHistory}
                                />

                                <div>
                                  <ActiveLearningStatus
                                    status={status}
                                    onStartNewBatch={handleStartNewBatch}
                                  />
                                </div>

                                <div className="space-y-4">
                                  <div className="flex gap-4">
                                    <Button
                                      onClick={async () => {
                                        try {
                                          // ALWAYS send current labels before export
                                          console.log(
                                            "Sending labels to backend before export:",
                                            labels,
                                          );

                                          if (labels.length > 0) {
                                            const labelResponse = await fetch(
                                              `http://localhost:8000/update-project-labels`,
                                              {
                                                method: "POST",
                                                headers: {
                                                  "Content-Type":
                                                    "application/json",
                                                },
                                                body: JSON.stringify({
                                                  labels: labels,
                                                }),
                                              },
                                            );

                                            if (!labelResponse.ok) {
                                              console.warn(
                                                "Failed to update labels, but continuing with export",
                                              );
                                            }
                                          }

                                          const response = await fetch(
                                            `http://localhost:8000/export-project`,
                                            {
                                              method: "GET",
                                            },
                                          );

                                          if (!response.ok) {
                                            throw new Error(
                                              `Failed to export project: ${response.statusText}`,
                                            );
                                          }

                                          const blob = await response.blob();
                                          const url =
                                            window.URL.createObjectURL(blob);
                                          const a = document.createElement("a");
                                          a.href = url;

                                          const contentDisposition =
                                            response.headers.get(
                                              "content-disposition",
                                            );
                                          let filename = `${projectName}_project_${new Date()
                                            .toISOString()
                                            .slice(0, 10)}.zip`;

                                          if (contentDisposition) {
                                            const filenameMatch =
                                              contentDisposition.match(
                                                /filename="([^"]+)"/,
                                              );
                                            if (filenameMatch) {
                                              filename = filenameMatch[1];
                                            }
                                          }

                                          a.download = filename;
                                          document.body.appendChild(a);
                                          a.click();
                                          window.URL.revokeObjectURL(url);
                                          document.body.removeChild(a);

                                          setSuccessMessage(
                                            "Project exported successfully!",
                                          );
                                        } catch (error) {
                                          setErrorMessage(
                                            "Failed to export project: " +
                                              error.message,
                                          );
                                        }
                                      }}
                                      disabled={!isInitialized || !projectName}
                                    >
                                      <Download className="h-4 w-4 mr-2" />
                                      Export Project with Latest Checkpoint and
                                      Labels
                                    </Button>
                                  </div>
                                </div>
                              </>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </TabsContent>
                </Tabs>
              </div>

              {/* Right Column - Image Annotation */}
              <div>
                <Card className="sticky top-6">
                  <CardHeader>
                    <CardTitle>Image Annotation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Image Display with Navigation */}
                      <div className="bg-gray-100 rounded-lg overflow-hidden">
                        <img
                          src={currentImage}
                          alt="Current image for annotation"
                          className="w-full object-contain max-h-96"
                        />
                        {loadedImages.length > 0 && (
                          <div className="p-4 bg-white border-t flex justify-between items-center">
                            <Button
                              onClick={handlePreviousImage}
                              disabled={currentImageIndex === 0}
                              variant="outline"
                            >
                              Previous
                            </Button>
                            <span className="text-sm text-gray-600">
                              Image {currentImageIndex + 1} of {batchSize}
                            </span>
                            <Button
                              onClick={handleNextImage}
                              disabled={
                                currentImageIndex === loadedImages.length - 1
                              }
                              variant="outline"
                            >
                              Next
                            </Button>
                          </div>
                        )}
                      </div>
                      {message.text &&
                        (() => {
                          const alertProps = getAlertProps(message.type);
                          const IconComponent = alertProps.icon;
                          return (
                            <Alert
                              variant={alertProps.variant}
                              className={alertProps.className}
                            >
                              <IconComponent className="h-4 w-4" />
                              <AlertDescription>
                                {message.text}
                              </AlertDescription>
                            </Alert>
                          );
                        })()}
                      {/* Label Selection */}
                      <div className="space-y-2">
                        <Label>Assign Label</Label>

                        <RadioGroup
                          value={selectedLabel}
                          onValueChange={setSelectedLabel}
                          className="space-y-2"
                        >
                          {labels.map((label, index) => (
                            <div
                              key={index}
                              className="flex items-center space-x-2"
                            >
                              <RadioGroupItem
                                value={index.toString()}
                                id={`label-${index}`}
                              />
                              <Label htmlFor={`label-${index}`}>{label}</Label>
                            </div>
                          ))}
                        </RadioGroup>
                      </div>
                      {/* Action Buttons */}
                      <div className="flex justify-between pt-4">
                        <Button
                          className="w-full"
                          onClick={handleSubmitLabel}
                          disabled={!selectedLabel || isRetraining}
                        >
                          {isRetraining ? (
                            <span className="flex items-center gap-2">
                              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                              Training in progress...
                            </span>
                          ) : (
                            "Submit Label"
                          )}
                        </Button>
                      </div>
                      {isRetraining && (
                        <div className="flex items-center gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
                          <div className="animate-spin rounded-full h-3 w-3 border-2 border-amber-600 border-t-transparent flex-shrink-0" />
                          Episode training running — label submission will
                          resume when complete.
                        </div>
                      )}

                      <BatchProgress
                        currentBatch={currentBatch}
                        batchStats={batchStats}
                        onSubmitLabel={handleSubmitLabel}
                        selectedLabel={selectedLabel}
                        isRetraining={isRetraining}
                        validationAccuracy={validationAccuracy}
                      />

                      {currentBatch.length > 0 &&
                        currentImageIndex < currentBatch.length &&
                        (() => {
                          const rawPredictions =
                            currentBatch[currentImageIndex]?.predictions || [];
                          const filteredPredictions = getFilteredPredictions(
                            rawPredictions,
                            labels,
                          );

                          console.log(
                            "=== CREATING NEW COMPONENT INSTANCE ===",
                          );
                          console.log(
                            "Predictions to render:",
                            filteredPredictions,
                          );

                          // Render inline to avoid any component caching issues
                          return (
                            <div
                              key={`predictions-${currentImageIndex}-${Date.now()}`}
                            >
                              <h4 className="font-medium mb-2">
                                Model Predictions
                              </h4>
                              <div className="text-xs text-gray-400 mb-2">
                                Image {currentImageIndex + 1} - Showing{" "}
                                {filteredPredictions.length} predictions
                              </div>
                              <div className="space-y-2">
                                {filteredPredictions.map((pred, idx) => {
                                  const confidence = Math.round(
                                    pred.confidence * 100,
                                  );
                                  console.log(
                                    `RENDERING: ${pred.label} with ${confidence}% (raw: ${pred.confidence})`,
                                  );

                                  return (
                                    <div
                                      key={`${pred.label}-${pred.confidence}-${idx}`}
                                      className="flex items-center gap-2"
                                    >
                                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                                        <div
                                          className="bg-blue-600 h-2.5 rounded-full"
                                          style={{
                                            width: `${Math.max(
                                              confidence,
                                              1,
                                            )}%`,
                                          }}
                                        ></div>
                                      </div>
                                      <span className="text-sm whitespace-nowrap">
                                        {pred.label}:{" "}
                                        {(pred.confidence * 100).toFixed(2)}%
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })()}

                      {isRetraining && (
                        <div className="mt-4">
                          <Card className="bg-blue-50 border-blue-200">
                            <CardContent className="p-4">
                              <div className="flex items-center space-x-2">
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                                <p className="text-blue-700">
                                  Training in progress... Please wait
                                </p>
                              </div>
                              <p className="text-sm text-blue-600 mt-2">
                                This may take several minutes depending on the
                                dataset size
                              </p>
                            </CardContent>
                          </Card>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default ActiveLearningUI;
