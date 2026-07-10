import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

const ValidationProgress = ({ status }) => {
  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle className="text-lg">Validation Set Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Total Validation Samples:</span>
            <span>{status.total}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Labeled:</span>
            <span>{status.labeled}</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${status.percent_labeled}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 text-center">
            {status.percent_labeled.toFixed(1)}% Complete
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ValidationProgress;
