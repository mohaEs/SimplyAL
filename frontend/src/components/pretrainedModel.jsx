import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Alert, AlertDescription } from "./ui/alert";
import { AlertCircle, CheckCircle, Upload } from "lucide-react";
import activeLearnAPI from "../services/activelearning";

const PretrainedModelImport = ({ onImportSuccess, onError }) => {
  const [modelFile, setModelFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState("resnet50");
  const [numClasses, setNumClasses] = useState(2);
  const [projectName, setProjectName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState({ text: null, type: "info" });

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check file extension
      const validExtensions = [".pt", ".pth", ".pkl", ".pickle"];
      const fileExtension = file.name
        .toLowerCase()
        .substring(file.name.lastIndexOf("."));

      if (validExtensions.includes(fileExtension)) {
        setModelFile(file);
        setMessage({ text: `Selected: ${file.name}`, type: "success" });
      } else {
        setMessage({
          text: "Please select a valid PyTorch model file (.pt, .pth, .pkl)",
          type: "error",
        });
        setModelFile(null);
      }
    }
  };

  const handleImportModel = async () => {
    if (!modelFile || !projectName) {
      setMessage({
        text: "Please select a model file and enter a project name",
        type: "error",
      });
      return;
    }

    try {
      setIsLoading(true);
      setMessage({ text: "Importing model...", type: "info" });

      const result = await activeLearnAPI.importPretrainedModel(
        modelFile,
        selectedModel,
        numClasses,
        projectName
      );

      setMessage({ text: "Model imported successfully!", type: "success" });

      // Call success callback with result
      if (onImportSuccess) {
        onImportSuccess(result);
      }
    } catch (error) {
      const errorMessage = error.message || "Failed to import model";
      setMessage({ text: errorMessage, type: "error" });

      if (onError) {
        onError(errorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyModel = async () => {
    if (!modelFile) {
      setMessage({ text: "Please select a model file first", type: "error" });
      return;
    }

    try {
      setIsLoading(true);
      setMessage({ text: "Analyzing model...", type: "info" });

      const result = await activeLearnAPI.verifyModelCompatibility(modelFile);

      if (result.compatible) {
        setMessage({
          text: `Model verified! Detected: ${
            result.model_type || "Unknown"
          } with ${result.num_classes || "Unknown"} classes`,
          type: "success",
        });

        // Auto-fill detected values
        if (result.model_type && result.model_type !== "unknown") {
          setSelectedModel(result.model_type);
        }
        if (result.num_classes) {
          setNumClasses(result.num_classes);
        }
      } else {
        setMessage({
          text: result.message || "Model verification failed",
          type: "error",
        });
      }
    } catch (error) {
      setMessage({
        text: "Error verifying model: " + error.message,
        type: "error",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getAlertProps = (messageType) => {
    switch (messageType) {
      case "success":
        return {
          className: "border-green-200 bg-green-50 text-green-800",
          icon: CheckCircle,
        };
      case "error":
        return {
          className: "border-red-200 bg-red-50 text-red-800",
          icon: AlertCircle,
        };
      default:
        return {
          className: "border-blue-200 bg-blue-50 text-blue-800",
          icon: AlertCircle,
        };
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Import Pretrained Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Project Name */}
        <div>
          <Label htmlFor="project-name">Project Name</Label>
          <Input
            id="project-name"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="Enter project name"
            disabled={isLoading}
          />
        </div>

        {/* Model Type Selection */}
        <div>
          <Label>Model Type</Label>
          <Select
            value={selectedModel}
            onValueChange={setSelectedModel}
            disabled={isLoading}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select model type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="resnet18">ResNet18</SelectItem>
              <SelectItem value="resnet50">ResNet50</SelectItem>
              <SelectItem value="dinov2">
                DINOv2 (ViT-B/14)
              </SelectItem>
              <SelectItem value="custom">Custom Model</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Number of Classes */}
        <div>
          <Label htmlFor="num-classes">Number of Classes</Label>
          <Input
            id="num-classes"
            type="number"
            value={numClasses}
            onChange={(e) => setNumClasses(parseInt(e.target.value) || 2)}
            min={2}
            disabled={isLoading}
          />
        </div>

        {/* File Upload */}
        <div>
          <Label htmlFor="model-file">Model File</Label>
          <div className="mt-2">
            <Input
              id="model-file"
              type="file"
              accept=".pt,.pth,.pkl,.pickle"
              onChange={handleFileChange}
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Custom Model Help Text */}
        {selectedModel === "custom" && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded">
            <p className="text-sm text-blue-700">
              <strong>Custom Model:</strong> Upload any PyTorch model file (.pt,
              .pth). The system will attempt to automatically detect the
              architecture and adapt it for your classes. You can verify the
              model first to see what's detected.
            </p>
          </div>
        )}

        {/* Status Message */}
        {message.text &&
          (() => {
            const alertProps = getAlertProps(message.type);
            const IconComponent = alertProps.icon;
            return (
              <Alert className={alertProps.className}>
                <IconComponent className="h-4 w-4" />
                <AlertDescription>{message.text}</AlertDescription>
              </Alert>
            );
          })()}

        {/* Action Buttons */}
        <div className="flex gap-2">
          {selectedModel === "custom" && modelFile && (
            <Button
              onClick={handleVerifyModel}
              disabled={isLoading}
              variant="outline"
              className="flex-1"
            >
              <Upload className="h-4 w-4 mr-2" />
              Verify Model
            </Button>
          )}

          <Button
            onClick={handleImportModel}
            disabled={isLoading || !modelFile || !projectName}
            className="flex-1"
          >
            {isLoading ? "Importing..." : "Import Model"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default PretrainedModelImport;
