import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const EnhancedMetricsDisplay = ({ metrics, episode_history, lr_history }) => {
  // Transform loss data directly from the API format
  const lossData =
    metrics?.current_epoch_losses?.x?.map((x, i) => ({
      epoch: x,
      loss: metrics.current_epoch_losses.y[i],
    })) ||
    metrics?.currentEpochLosses?.x?.map((x, i) => ({
      epoch: x,
      loss: metrics.currentEpochLosses.y[i],
    })) ||
    [];

  // Transform accuracy data
  const accuracyData =
    metrics?.episode_accuracies?.x?.map((x, i) => ({
      episode: x,
      accuracy: metrics.episode_accuracies.y[i],
    })) ||
    metrics?.episodeAccuracies?.x?.map((x, i) => ({
      episode: x,
      accuracy: metrics.episodeAccuracies.y[i],
    })) ||
    [];

  // Get current episode and best accuracy from multiple possible sources
  const currentEpisode = (() => {
    // First try to get from metrics (most reliable)
    if (metrics?.current_episode !== undefined) {
      return metrics.current_episode;
    }

    // If we have episode history, the current episode is the next one to run
    if (episode_history?.length > 0) {
      const lastCompletedEpisode = Math.max(
        ...episode_history.map((ep) => ep.episode || 0)
      );
      return lastCompletedEpisode + 1;
    }

    // If we have accuracy data, get the next episode
    if (accuracyData?.length > 0) {
      const lastEpisode = Math.max(...accuracyData.map((d) => d.episode));
      return lastEpisode + 1;
    }

    // Default to episode 1 (first episode to run)
    return 1;
  })();

  const bestValidationAccuracy = (() => {
    // First check if we have explicit best_val_acc from metrics
    if (metrics?.best_val_acc && metrics.best_val_acc > 0) {
      return metrics.best_val_acc;
    }

    // Check episode history for best accuracy
    if (episode_history?.length > 0) {
      const validAccuracies = episode_history
        .map((ep) => ep.best_val_acc || ep.validation_accuracy || 0)
        .filter((acc) => acc > 0);

      if (validAccuracies.length > 0) {
        return Math.max(...validAccuracies);
      }
    }

    // Check accuracy data array
    if (accuracyData?.length > 0) {
      const validAccuracies = accuracyData
        .map((d) => d.accuracy)
        .filter((acc) => acc > 0);

      if (validAccuracies.length > 0) {
        return Math.max(...validAccuracies);
      }
    }

    // Final fallback
    return 0;
  })();

  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle>Training Metrics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-8">
        {/* Current Episode and Best Accuracy */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
          <div>
            <h3 className="font-medium text-sm text-gray-600">
              Current Episode
            </h3>
            <p className="text-2xl font-bold">{currentEpisode}</p>
          </div>
          <div>
            <h3 className="font-medium text-sm text-gray-600">
              Best Validation Accuracy
            </h3>
            <p className="text-2xl font-bold">
              {bestValidationAccuracy.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Training Loss Chart */}
        {lossData.length > 0 && (
          <div className="h-80">
            <h3 className="font-medium mb-4">Training Loss</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={lossData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "bottom" }}
                />
                <YAxis
                  label={{
                    value: "Loss",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#82ca9d"
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Episode Accuracy Chart */}
        {accuracyData.length > 0 && (
          <div className="h-80">
            <h3 className="font-medium mb-4">Episode Accuracy</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={accuracyData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="episode"
                  label={{ value: "Episode", position: "bottom" }}
                />
                <YAxis
                  domain={[0, 100]}
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#8884d8"
                  name="Validation Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Learning Rate Chart */}
        {lr_history && lr_history.length > 0 && (
          <div className="h-80">
            <h3 className="font-medium mb-4">Learning Rate</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={lr_history}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "bottom" }}
                />
                <YAxis
                  scale="log"
                  label={{
                    value: "Learning Rate",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="new_lr"
                  stroke="#8884d8"
                  name="Learning Rate"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Episode History */}
        {episode_history?.length > 0 && (
          <div className="mt-4">
            <h3 className="font-medium mb-2">Episode History</h3>
            <div className="space-y-2">
              {episode_history.map((episode, idx) => (
                <div key={idx} className="text-sm">
                  <div className="flex justify-between">
                    <span>
                      Episode {episode.episode ? episode.episode : "0"}
                    </span>
                    <span>
                      Best Acc:{" "}
                      {episode.best_val_acc ? episode.best_val_acc : "0.0"}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default EnhancedMetricsDisplay;
