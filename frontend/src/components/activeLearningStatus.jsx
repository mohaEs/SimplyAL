import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

const ActiveLearningStatus = ({ status }) => {
  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle>Active Learning Status</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Current Episode:</span>
            <span>{status.current_episode}</span>
          </div>
          <div className="flex justify-between">
            <span>Labeled Images (excluding validation):</span>
            <span>{status.labeled_count}</span>
          </div>
          <div className="flex justify-between">
            <span>Unlabeled Images:</span>
            <span>{status.unlabeled_count}</span>
          </div>
          <div className="flex justify-between">
            <span>Validation Set:</span>
            <span>{status.validation_count}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ActiveLearningStatus;
