import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";

// Enhanced validation progress component
const ValidationProgress = ({ status }) => {
  const percentage = status?.percent_labeled || 0;

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Validation Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Labeled Validation Samples:</span>
            <span>
              {status?.labeled || 0} / {status?.total || 0}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${percentage}%` }}
            />
          </div>
          <p className="text-sm text-gray-500">
            {percentage.toFixed(1)}% of validation set labeled
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

// Enhanced batch progress component
const BatchProgress = ({
  currentBatch,
  batchStats,
  onSubmitLabel,
  selectedLabel,
  isRetraining,
  validationAccuracy,
}) => {
  const progress = (batchStats.completed / batchStats.totalImages) * 100;

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Current Batch Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between text-sm">
            <span>Images Labeled:</span>
            <span>
              {batchStats.completed} / {batchStats.totalImages}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-green-600 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>

          {validationAccuracy > 0 && (
            <div className="text-sm text-gray-600">
              Latest Validation Accuracy: {validationAccuracy.toFixed(2)}%
            </div>
          )}

          {isRetraining && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Training in progress... Please wait
              </AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Enhanced model predictions component
const ModelPredictions = ({ predictions, labels }) => {
  if (!predictions || predictions.length === 0) return null;

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Model Predictions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {predictions.map((pred, idx) => (
            <div key={idx} className="flex justify-between items-center">
              <span>{labels[idx] || `Class ${idx}`}</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
                <span className="text-sm">
                  {(pred.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

const ActiveLearningStatus = ({ status, onStartNewBatch }) => {
  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Active Learning Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium">Current Episode</p>
              <p className="text-2xl">{status?.current_episode || 0}</p>
            </div>
            <div>
              <p className="text-sm font-medium">Labeled Images (excluding validation)</p>
              <p className="text-2xl">{status?.labeled_count || 0}</p>
            </div>
            <div>
              <p className="text-sm font-medium">Remaining Unlabeled</p>
              <p className="text-2xl">{status?.unlabeled_count || 0}</p>
            </div>
            <div>
              <p className="text-sm font-medium">Validation Set</p>
              <p className="text-2xl">{status?.validation_count || 0}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export {
  ValidationProgress,
  BatchProgress,
  ModelPredictions,
  ActiveLearningStatus,
};
