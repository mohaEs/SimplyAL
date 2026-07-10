import React from "react";
import { Alert, AlertDescription } from "./ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Label } from "./ui/label";
import { Loader2 } from "lucide-react";
import PerformanceDashboard from "./performanceDashboard";

const AutomatedTrainingControls = ({ status }) => {
  const getTrainingPhase = () => {
    if (!status?.is_training) return null;
    if (status?.current_batch?.labeled < status?.current_batch?.total) {
      return "labeling";
    }
    return "training";
  };

  const phase = getTrainingPhase();

  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle>Active Learning Progress</CardTitle>
      </CardHeader>
      <CardContent>
        {status && (
          <div className="space-y-4">
            {phase && (
              <Alert variant={phase === "training" ? "default" : "secondary"}>
                <div className="flex items-center gap-2">
                  {phase === "training" && (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  )}
                  <AlertDescription>
                    {phase === "labeling"
                      ? "Label all images in the current batch to continue training"
                      : "Training in progress — please wait..."}
                  </AlertDescription>
                </div>
              </Alert>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Current Episode</Label>
                <div className="text-2xl font-semibold">
                  {status.current_episode}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Labeled Images (excluding validation)</Label>
                <div className="text-lg">{status.labeled_count}</div>
              </div>
              <div>
                <Label>Remaining Unlabeled</Label>
                <div className="text-lg">{status.unlabeled_count}</div>
              </div>
            </div>

            <div>
              <PerformanceDashboard />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AutomatedTrainingControls;
