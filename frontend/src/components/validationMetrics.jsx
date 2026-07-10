import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

function ValidationMetrics({ validationMetrics }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Validation Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>Best Validation Accuracy</Label>
            <div className="text-2xl font-bold">
              {validationMetrics?.best_val_acc?.toFixed(2)}%
            </div>
          </div>
          <div>
            <Label>Current Validation Accuracy</Label>
            <div className="text-2xl">
              {validationMetrics?.current_val_acc?.toFixed(2)}%
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ValidationMetrics;
