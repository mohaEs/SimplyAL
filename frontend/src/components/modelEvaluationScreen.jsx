import React, { useState } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Download, ArrowRight, CheckCircle } from "lucide-react";

const ModelEvaluationScreen = ({
  evaluationData,
  onContinueTraining,
  onExportModel,
  labels = [],
  currentEpisode = 0,
}) => {
  const [isExporting, setIsExporting] = useState(false);

  if (!evaluationData || !evaluationData.predictions) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Model Evaluation</CardTitle>
        </CardHeader>
        <CardContent>
          <p>No evaluation data available.</p>
        </CardContent>
      </Card>
    );
  }

  const { predictions, overall_confidence, episode_info } = evaluationData;

  const handleExport = async () => {
    try {
      setIsExporting(true);
      await onExportModel();
    } catch (error) {
      console.error("Export failed:", error);
    } finally {
      setIsExporting(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return "text-green-600 bg-green-50";
    if (confidence >= 0.6) return "text-yellow-600 bg-yellow-50";
    return "text-red-600 bg-red-50";
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Medium";
    return "Low";
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-6 w-6 text-green-500" />
            Episode {currentEpisode} Complete - Model Performance Preview (top 10 uncertain samples from unlabled set)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {predictions.length}
              </div>
              <div className="text-sm text-gray-600">Images Evaluated</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {(overall_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Average Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {episode_info?.validation_accuracy
                  ? `${episode_info.validation_accuracy.toFixed(1)}%`
                  : "N/A"}
              </div>
              <div className="text-sm text-gray-600">Validation Accuracy</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Predictions Grid */}
      <Card>
        <CardHeader>
          <CardTitle>Model Predictions on Next 10 Unlabeled Images</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            {predictions.slice(0, 10).map((pred, index) => (
              <div key={pred.image_id} className="space-y-3">
                {/* Image */}
                <div className="relative">
                  <img
                    src={`http://localhost:8000/image/${pred.image_id}`}
                    alt={`Prediction ${index + 1}`}
                    className="w-full h-full object-cover rounded-lg border-2 border-gray-200"
                    onError={(e) => {
                      e.target.style.display = "none";
                      e.target.nextSibling.style.display = "flex";
                    }}
                  />
                  <div
                    className="hidden w-full h-32 bg-gray-200 rounded-lg items-center justify-center"
                    style={{ display: "none" }}
                  >
                    <span className="text-gray-500 text-sm">
                      Image not found
                    </span>
                  </div>
                </div>

                {/* Prediction Info */}
                <div className="space-y-2">
                  <div className="text-sm font-semibold text-center">
                    Image {index + 1}
                  </div>

                  {/* Top Prediction */}
                  <div className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-medium">
                        {labels[pred.predicted_class] ||
                          `Class ${pred.predicted_class}`}
                      </span>
                      <span
                        className={`text-xs px-2 py-1 rounded ${getConfidenceColor(
                          pred.confidence
                        )}`}
                      >
                        {getConfidenceLabel(pred.confidence)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${pred.confidence * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-center text-gray-600">
                      {(pred.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* All Class Probabilities */}
                  {pred.all_probabilities && (
                    <div className="space-y-1">
                      <div className="text-xs font-medium text-gray-700">
                        All Classes:
                      </div>
                      {pred.all_probabilities.map((prob, idx) => (
                        <div key={idx} className="flex justify-between text-xs">
                          <span
                            className={
                              prob.class_index === pred.predicted_class
                                ? "font-bold"
                                : ""
                            }
                          >
                            {labels[prob.class_index] ||
                              `Class ${prob.class_index}`}
                          </span>
                          <span
                            className={
                              prob.class_index === pred.predicted_class
                                ? "font-bold"
                                : ""
                            }
                          >
                            {(prob.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Confidence Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Confidence Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                label: "High Confidence (≥80%)",
                color: "green",
                count: predictions.filter((p) => p.confidence >= 0.8).length,
              },
              {
                label: "Medium Confidence (60-80%)",
                color: "yellow",
                count: predictions.filter(
                  (p) => p.confidence >= 0.6 && p.confidence < 0.8
                ).length,
              },
              {
                label: "Low Confidence (<60%)",
                color: "red",
                count: predictions.filter((p) => p.confidence < 0.6).length,
              },
            ].map((item) => (
              <div key={item.label} className="text-center">
                <div className={`text-2xl font-bold text-${item.color}-600`}>
                  {item.count}
                </div>
                <div className="text-sm text-gray-600">{item.label}</div>
                <div className="text-xs text-gray-500">
                  (
                  {predictions.length > 0
                    ? ((item.count / predictions.length) * 100).toFixed(1)
                    : 0}
                  %)
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Class Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Predicted Class Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            {labels.map((label, index) => {
              const count = predictions.filter(
                (p) => p.predicted_class === index
              ).length;
              const percentage =
                predictions.length > 0 ? (count / predictions.length) * 100 : 0;

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded"
                >
                  <span className="font-medium">{label}</span>
                  <div className="text-right">
                    <div className="font-bold">{count}</div>
                    <div className="text-sm text-gray-600">
                      {percentage.toFixed(1)}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex justify-center space-x-4">
            <Button
              onClick={onContinueTraining}
              size="lg"
              className="flex items-center gap-2"
            >
              <ArrowRight className="h-4 w-4" />
              Continue
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelEvaluationScreen;
